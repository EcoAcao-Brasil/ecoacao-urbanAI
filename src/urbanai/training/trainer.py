"""
Training for UrbanAI Models

Handles complete training workflow with callbacks and logging.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from urbanai.models import ConvLSTMEncoderDecoder
from urbanai.training.dataset import create_dataloaders
from urbanai.training.losses import SpatialMSELoss
from urbanai.training.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class UrbanAITrainer:
    """
    Training orchestrator for UrbanAI models.

    Args:
        data_dir: Directory with processed features
        output_dir: Directory for model checkpoints and logs
        config: Training configuration
        device: Device to train on ('cuda' or 'cpu')
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

        self.config = config or self._default_config()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = self._build_model()
        self.model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = self._build_optimizer()
        self.criterion = self._build_criterion()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rates": [],
        }

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.output_dir / "tensorboard")

        logger.info("UrbanAI Trainer initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 8,
        learning_rate: Optional[float] = None,
        early_stopping_patience: int = 15,
        save_every: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute training loop.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate (overrides config)
            early_stopping_patience: Patience for early stopping
            save_every: Save checkpoint every N epochs

        Returns:
            Training history and final metrics
        """
        logger.info("=" * 80)
        logger.info("TRAINING START")
        logger.info("=" * 80)

        # Update config if parameters provided
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            data_dir=self.data_dir,
            sequence_length=self.config["sequence_length"],
            prediction_horizon=self.config["prediction_horizon"],
            batch_size=batch_size,
            num_workers=self.config.get("num_workers", 4),
            normalize=True,
            augment_train=True,
        )

        # Early stopping
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self._validate_epoch(val_loader)

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["learning_rates"].append(self.optimizer.param_groups[0]["lr"])

            # Logging
            self._log_epoch(epoch, epochs, train_loss, val_loss, val_metrics)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best")
                patience_counter = 0
                logger.info(f"✓ New best model saved (val_loss: {val_loss:.6f})")
            else:
                patience_counter += 1

            # Regular checkpoint
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Save final model
        self._save_checkpoint("last")

        # Close tensorboard
        self.writer.close()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 80)

        return {
            "history": self.history,
            "best_loss": self.best_val_loss,
            "final_epoch": self.current_epoch,
        }

    def _train_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs, future_steps=targets.shape[1])

            # Calculate loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["gradient_clip"],
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        return avg_loss

    def _validate_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
    ) -> tuple[float, Dict[str, float]]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")

            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs, future_steps=targets.shape[1])

                # Calculate loss
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                # Store for metrics
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

                # Update progress
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.6f}"})

        # Calculate metrics
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
        """Log epoch results."""
        # Console
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"MAE: {metrics['mae']:.6f} | "
            f"RMSE: {metrics['rmse']:.6f}"
        )

        # Tensorboard
        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Loss/val", val_loss, epoch)
        self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)

    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"convlstm_{name}.pth"

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def _build_model(self) -> nn.Module:
        """Build model from config."""
        model_config = self.config.get("model", {})

        model = ConvLSTMEncoderDecoder(
            input_channels=model_config.get("input_channels", 7),
            hidden_dims=model_config.get("hidden_dims", [64, 128, 256, 256, 128, 64]),
            kernel_size=(model_config.get("kernel_size", 3), model_config.get("kernel_size", 3)),
            num_encoder_layers=model_config.get("num_encoder_layers", 3),
            num_decoder_layers=model_config.get("num_decoder_layers", 3),
            output_channels=model_config.get("output_channels", 7),
        )

        return model

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)

        if optimizer_name == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_name == "adamw":
            optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _build_criterion(self) -> nn.Module:
        """Build loss function from config."""
        loss_name = self.config.get("loss", "mse").lower()

        if loss_name == "mse":
            criterion = nn.MSELoss()
        elif loss_name == "spatial_mse":
            criterion = SpatialMSELoss()
        elif loss_name == "mae":
            criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        return criterion

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default training configuration."""
        return {
            "sequence_length": 10,
            "prediction_horizon": 1,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "loss": "mse",
            "gradient_clip": 1.0,
            "num_workers": 4,
            "model": {
                "input_channels": 7,
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
                "num_encoder_layers": 3,
                "num_decoder_layers": 3,
                "output_channels": 7,
            },
        }
