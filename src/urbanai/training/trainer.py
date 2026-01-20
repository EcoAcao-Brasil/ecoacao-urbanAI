"""
Training Engine

Manages the full training workflow for UrbanAI models
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from urbanai.models import ConvLSTMEncoderDecoder
from urbanai.training.dataset import create_dataloaders
from urbanai.training.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class UrbanAITrainer:
    """
    Orchestrates the model training process.

    This class abstracts away the PyTorch boilerplate, handling device
    management, gradient updates, metric tracking (TensorBoard/Console),
    and checkpointing with metadata.
    """

    def __init__(
        self,
        data_dir: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cuda",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Merge user config with defaults, prioritizing user values
        default_config = self._default_config()
        if config:
            self.config = self._deep_merge_config(default_config, config)
        else:
            self.config = default_config

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the ConvLSTM architecture
        self.model = self._build_model()
        self.model.to(self.device)

        self.optimizer = self._build_optimizer()

        # Standard MSE is sufficient for pixel-level temperature regression
        self.criterion = nn.MSELoss()

        # State tracking
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }

        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")

        logger.info("UrbanAI Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Log gradient accumulation if enabled
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)
        if grad_accum_steps > 1:
            logger.info(f"Gradient accumulation: {grad_accum_steps} steps")

        # Log effective configuration
        logger.info("Effective training configuration:")
        train_cfg = self.config.get("training", {})
        logger.info(
            f"  Sequence Length: {train_cfg.get('sequence_length', self.config.get('sequence_length', 10))}"
        )
        logger.info(
            f"  Prediction Horizon: {train_cfg.get('prediction_horizon', self.config.get('prediction_horizon', 1))}"
        )
        logger.info(
            f"  Batch Size: {train_cfg.get('batch_size', self.config.get('batch_size', 8))}"
        )
        logger.info(f"  Learning Rate: {self.config.get('learning_rate', 0.001)}")

        # Log effective batch size if using gradient accumulation
        grad_accum = self.config.get("gradient_accumulation_steps", 1)
        if grad_accum > 1:
            physical_batch = train_cfg.get('batch_size', self.config.get('batch_size', 8))
            effective_batch = physical_batch * grad_accum
            logger.info(f"  Gradient Accumulation: {grad_accum} steps")
            logger.info(f"  Effective Batch Size: {effective_batch} (physical: {physical_batch})")

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: Optional[float] = None,
        early_stopping_patience: int = 15,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes the main training loop.
        """
        logger.info("Starting training...")

        # Allow runtime override of LR without rebuilding the optimizer
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

        train_loader, val_loader = create_dataloaders(
            data_dir=self.data_dir,
            config=self.config,
        )

        # Capture dataset reference to extract normalization stats later
        train_dataset = train_loader.dataset

        patience_counter = 0

        for epoch in range(epochs):
            self.current_epoch = epoch

            # Run cycles
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate_epoch(val_loader)

            # Record metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            self._log_epoch(epoch, epochs, train_loss, val_loss, val_metrics)

            # Checkpointing logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Pass dataset to save stats
                self._save_checkpoint("best", train_dataset=train_dataset)
                patience_counter = 0
                logger.info(f"New best model (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}", train_dataset=train_dataset)

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Ensure we always have the final state saved with stats
        self._save_checkpoint("last", train_dataset=train_dataset)
        self.writer.close()

        logger.info("Training completed")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")

        return {
            "history": self.history,
            "best_loss": self.best_val_loss,
            "final_epoch": self.current_epoch,
        }

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Runs a single pass over the training set."""
        self.model.train()
        total_loss = 0.0
        
        # Get gradient accumulation steps
        grad_accum_steps = self.config.get("gradient_accumulation_steps", 1)

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs, future_steps=targets.shape[1])
            loss = self.criterion(outputs, targets)
            
            # Scale loss by accumulation steps
            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

            # Backward pass
            loss.backward()

            # Only update weights every N steps
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Gradient clipping
                if self.config.get("gradient_clip"):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["gradient_clip"],
                    )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate unscaled loss for logging
            total_loss += loss.item() * grad_accum_steps
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        return avg_loss

    def _validate_epoch(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluates the model on the validation set without gradient tracking."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs, future_steps=targets.shape[1])
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Accumulate for aggregate metric calculation (MAE, RMSE)
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_outputs, all_targets)

        return avg_loss, metrics

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float],
    ) -> None:
        """Updates both the console logger and TensorBoard."""
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"MAE: {metrics['mae']:.6f} | "
            f"RMSE: {metrics['rmse']:.6f}"
        )

        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    def _save_checkpoint(self, name: str, train_dataset: Optional[Dataset] = None) -> None:
        """
        Serializes model state, optimizer state, and history to disk.
        Includes normalization statistics to enable correct denormalization during inference.
        """
        checkpoint_path = self.output_dir / f"convlstm_{name}.pth"

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
            # Placeholders for normalization info
            "normalization_stats": None,
            "normalization_method": None,
        }

        # Embed normalization stats if dataset is available
        if train_dataset is not None:
            # Assumes dataset has a method to retrieve computed stats
            if hasattr(train_dataset, "get_normalization_stats"):
                norm_stats = train_dataset.get_normalization_stats()
                if norm_stats is not None:
                    checkpoint["normalization_stats"] = norm_stats

                    # Retrieve method from config or default to 'zscore'
                    train_conf = self.config.get("training", {})
                    checkpoint["normalization_method"] = train_conf.get(
                        "normalization_method", "zscore"
                    )

                    logger.debug(
                        f"Embedded normalization stats ({checkpoint['normalization_method']}) into checkpoint."
                    )

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _build_model(self) -> nn.Module:
        """Instantiates the ConvLSTM based on the config dictionary."""
        model_config = self.config.get("model", {})

        model = ConvLSTMEncoderDecoder(
            input_channels=model_config.get("input_channels", 5),
            hidden_dims=model_config.get("hidden_dims", [64, 128, 256, 256, 128, 64]),
            kernel_size=(model_config.get("kernel_size", 3), model_config.get("kernel_size", 3)),
            num_encoder_layers=model_config.get("num_encoder_layers", 3),
            num_decoder_layers=model_config.get("num_decoder_layers", 3),
            output_channels=model_config.get("output_channels", 5),
        )

        return model

    def _build_optimizer(self) -> optim.Optimizer:
        """Constructs the optimizer."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)

        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    @staticmethod
    def _deep_merge_config(default: Dict, user: Dict) -> Dict:
        """
        Recursively merge user config into defaults, prioritizing user values.

        Args:
            default: Default configuration dictionary
            user: User-provided configuration dictionary

        Returns:
            Merged configuration with user values taking precedence
        """
        merged = default.copy()
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = UrbanAITrainer._deep_merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Provides sensible defaults for sequence-to-sequence heat prediction."""
        return {
            "sequence_length": 10,
            "prediction_horizon": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "gradient_clip": 1.0,
            "gradient_accumulation_steps": 1,
            "num_workers": 4,
            "training": {"normalization_method": "zscore"},
            "model": {
                "input_channels": 5,  # Default to 5 (no Tocantins)
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "output_channels": 5,  # Default to 5 (no Tocantins)
            },
        }
