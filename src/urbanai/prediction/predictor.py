"""
Future Prediction Engine

Generates future urban heat predictions using trained models.
"""

import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
import torch
import torch.nn as nn
from tqdm import tqdm

# Assumes local project structure; generic import used for demonstration
from urbanai.models import ConvLSTMEncoderDecoder

logger = logging.getLogger(__name__)


class FuturePredictor:
    """
    Predicts future urban heat landscapes using trained ConvLSTM models.

    Implements autoregressive prediction for multi-step forecasting and 
    Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        data_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
    ) -> None:
        """
        Initialize the predictor.

        Args:
            model_path: Path to the .pt model checkpoint.
            data_dir: Directory containing processed feature rasters.
            output_dir: Directory to save results (defaults to sibling 'predictions' dir).
            device: Compute device ('cuda', 'mps', or 'cpu').
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialization
        self.model, self.config, self.norm_stats, self.norm_method = self._load_model()
        self.model.eval()
        
        # Determine band names based on model configuration
        self.band_names = self._get_band_names()

        logger.info(f"FuturePredictor ready on {self.device}")
        logger.info(f"Model channels: {self.config.get('model', {}).get('input_channels', 5)}")
        logger.info(f"Band names: {self.band_names}")
        logger.info(f"Normalization: {self.norm_method.upper() if self.norm_stats else 'DISABLED'}")

    def predict(
        self,
        current_year: int,
        target_year: int,
        save_outputs: bool = True,
        calculate_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute prediction pipeline from current_year to target_year.

        Args:
            current_year: The last available year of ground truth data.
            target_year: The future year to predict.
            save_outputs: Whether to write GeoTIFFs to disk.
            calculate_uncertainty: Whether to estimate epistemic uncertainty.

        Returns:
            Dict containing metadata, statistics, and paths to saved files.
        """
        logger.info(f"Starting prediction task: {current_year} -> {target_year}")

        # Validation and Setup
        year_interval = self._infer_interval()
        if target_year <= current_year:
            raise ValueError(f"Target year ({target_year}) must be greater than current year ({current_year})")
        
        n_steps = (target_year - current_year) // year_interval
        logger.info(f"Horizon: {n_steps} steps (Interval: {year_interval} years)")

        # Prepare Input
        input_sequence = self._load_input_sequence(current_year)
        
        # Inference
        prediction = self._predict_autoregressive(input_sequence, n_steps)

        # Post-processing
        if self.norm_stats:
            logger.info("Denormalizing predictions...")
            prediction = self._denormalize(prediction)

        # Uncertainty Estimation
        uncertainty = None
        if calculate_uncertainty:
            uncertainty = self._estimate_uncertainty(input_sequence, n_steps)
            if self.norm_stats and uncertainty is not None:
                uncertainty = self._denormalize_uncertainty(uncertainty)

        # Output Generation
        result_meta = {
            "target_year": target_year,
            "prediction_shape": prediction.shape,
            "statistics": self._calculate_prediction_stats(prediction),
            "normalization_method": self.norm_method,
        }

        if save_outputs:
            ref_meta = self._get_raster_metadata(current_year)
            out_path = self._save_prediction(prediction, ref_meta, target_year, uncertainty)
            result_meta["output_path"] = str(out_path)

        logger.info(f"Prediction task complete for {target_year}")
        return result_meta

    def _get_band_names(self) -> List[str]:
        """
        Get band names based on model configuration.
        
        Returns:
            List of band names matching the model's channel count
        """
        n_channels = self.config.get("model", {}).get("input_channels", 5)
        base_bands = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]
        
        if n_channels == 7:
            return base_bands + ["IS", "SS"]
        return base_bands

    def _load_model(self) -> Tuple[nn.Module, Dict, Optional[Dict], str]:
        """Load checkpoint, configuration, and normalization stats."""
        logger.info(f"Loading checkpoint: {self.model_path.name}")
        checkpoint = torch.load(self.model_path, map_location=self.device)

        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        
        # Load Normalization Stats
        norm_stats = checkpoint.get("normalization_stats")
        norm_method = checkpoint.get("normalization_method", "zscore")
        
        if norm_stats:
            # Ensure stats are numpy arrays for broadcasting
            norm_stats = {
                k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for k, v in norm_stats.items()
            }
        else:
            logger.warning("⚠️ Checkpoint lacks normalization stats. Predictions may be unscaled.")

        # Initialize Model - use 5 as default for backward compatibility
        model = ConvLSTMEncoderDecoder(
            input_channels=model_config.get("input_channels", 5),
            hidden_dims=model_config.get("hidden_dims", [64, 128, 256, 256, 128, 64]),
            kernel_size=model_config.get("kernel_size", 3),
            num_encoder_layers=model_config.get("num_encoder_layers", 3),
            num_decoder_layers=model_config.get("num_decoder_layers", 3),
            output_channels=model_config.get("output_channels", 5),
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        return model, config, norm_stats, norm_method

    def _load_input_sequence(self, end_year: int) -> torch.Tensor:
        """Load and normalize historical sequence ending at end_year."""
        seq_len = self.config.get("sequence_length", 10)
        interval = self._infer_interval()
        
        start_year = end_year - ((seq_len - 1) * interval)
        years = list(range(start_year, end_year + 1, interval))

        logger.debug(f"Loading sequence years: {years}")

        seq_data = []
        for y in years:
            path = self._find_file_for_year(y)
            with rasterio.open(path) as src:
                data = src.read()
                # Sanitize input
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                seq_data.append(data)

        # Shape: (seq_len, C, H, W)
        sequence = np.stack(seq_data, axis=0)

        if self.norm_stats:
            sequence = self._normalize(sequence)

        # Convert to Tensor: (1, seq_len, C, H, W)
        tensor = torch.from_numpy(sequence).float().unsqueeze(0)
        return tensor.to(self.device)

    @torch.inference_mode()
    def _predict_autoregressive(self, input_seq: torch.Tensor, steps: int) -> np.ndarray:
        """
        Perform multi-step autoregressive prediction.
        
        Uses the output of step T as the input for step T+1.
        """
        curr_seq = input_seq.clone()
        final_pred = None

        for _ in tqdm(range(steps), desc="Forecasting", leave=False):
            # Model output shape: (1, 1, C, H, W)
            next_step = self.model(curr_seq, future_steps=1)
            
            # Slide window: Drop oldest time step, append prediction
            curr_seq = torch.cat([curr_seq[:, 1:], next_step], dim=1)
            final_pred = next_step

        # Return result of the final step: (C, H, W)
        return final_pred[0, 0].cpu().numpy()

    def _estimate_uncertainty(
        self, 
        input_seq: torch.Tensor, 
        steps: int, 
        mc_samples: int = 20
    ) -> Optional[np.ndarray]:
        """Estimate uncertainty using Monte Carlo Dropout."""
        logger.info(f"Calculating uncertainty ({mc_samples} MC samples)...")
        
        preds = []
        
        # Context manager to enable dropout during inference
        with self._enable_dropout():
            try:
                preds = []
                for _ in tqdm(range(mc_samples)):
                    with torch.no_grad():
                        pred = self._predict_single_sample(input_seq, steps)
                        preds.append(pred)
                        torch.cuda.empty_cache()
                
                stack = np.stack(preds, axis=0)  # (samples, C, H, W)
                std_dev = np.std(stack, axis=0)  # (C, H, W)
                return np.mean(std_dev, axis=0)  # Average across channels -> (H, W)

            except Exception as e:
                logger.error(f"Uncertainty calculation failed: {e}")
                return None
                
    def _predict_single_sample(self, input_seq: torch.Tensor, steps: int) -> np.ndarray:
        """
        Single forward pass for MC Dropout uncertainty estimation.
        
        Args:
            input_seq: Input sequence tensor
            steps: Number of future steps
            
        Returns:
            Prediction array
        """
        return self._predict_autoregressive(input_seq, steps)
          
    @contextmanager
    def _enable_dropout(self):
        """Context manager to enable Dropout layers while keeping Batch Norm frozen."""
        # Save original state
        original_mode = self.model.training
        
        # Set to train to enable dropout
        self.model.train()
        
        # Optional: Freeze BatchNorm layers to prevent stats update during inference
        # This is strictly more correct for MC Dropout than just .train()
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()
                
        try:
            yield
        finally:
            self.model.train(original_mode)

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization with automatic broadcasting."""
        if not self.norm_stats:
            return data

        # Prepare shapes for broadcasting: (C, 1, 1) or (1, C, 1, 1)
        shape_suffix = (1, 1) if data.ndim == 3 else (1, 1, 1)
        
        stats = self._get_reshaped_stats(shape_suffix)
        
        if self.norm_method == "zscore":
            return (data - stats["mean"]) / stats["std"]
        elif self.norm_method == "minmax":
            return (data - stats["min"]) / stats["range"]
        elif self.norm_method == "robust":
            return (data - stats["median"]) / stats["iqr"]
        return data

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if not self.norm_stats:
            return data

        shape_suffix = (1, 1) if data.ndim == 3 else (1, 1, 1)
        stats = self._get_reshaped_stats(shape_suffix)

        if self.norm_method == "zscore":
            return (data * stats["std"]) + stats["mean"]
        elif self.norm_method == "minmax":
            return (data * stats["range"]) + stats["min"]
        elif self.norm_method == "robust":
            return (data * stats["iqr"]) + stats["median"]
        return data

    def _denormalize_uncertainty(self, uncertainty: np.ndarray) -> np.ndarray:
        """Scale uncertainty map (Standard Deviation) back to original units."""
        # For std dev, we only scale by the multiplicative factor (Std, Range, IQR)
        # We do NOT add the mean/min/median.
        
        # Average the scaling factor across channels since uncertainty is flattened to (H, W)
        if self.norm_method == "zscore":
            scale = np.mean(self.norm_stats["std"])
        elif self.norm_method == "minmax":
            scale = np.mean(self.norm_stats["range"])
        elif self.norm_method == "robust":
            scale = np.mean(self.norm_stats["iqr"])
        else:
            scale = 1.0
            
        return uncertainty * scale

    def _get_reshaped_stats(self, suffix_shape: Tuple) -> Dict[str, np.ndarray]:
        """Helper to reshape stats for broadcasting."""
        reshaped = {}
        for k, v in self.norm_stats.items():
            # Add dimensions to the right: (C,) -> (C, 1, 1)
            target_shape = v.shape + suffix_shape
            reshaped[k] = v.reshape(target_shape)
            
            # If input is 4D (Time, C, H, W), we need (1, C, H, W) alignment
            if len(suffix_shape) == 3: 
                reshaped[k] = reshaped[k][None, ...] 
        return reshaped

    def _save_prediction(
        self, 
        pred: np.ndarray, 
        meta: Dict, 
        year: int, 
        uncertainty: Optional[np.ndarray]
    ) -> Path:
        """Write prediction and optional uncertainty to GeoTIFF."""
        meta.update({"count": pred.shape[0], "dtype": "float32"})
        
        out_path = self.output_dir / f"{year}_predicted.tif"
        
        # Save Prediction
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(pred.astype(np.float32))
            for i, name in enumerate(self.band_names, start=1):
                dst.set_band_description(i, name)

        # Save Uncertainty (if exists)
        if uncertainty is not None:
            unc_path = self.output_dir / f"{year}_uncertainty.tif"
            unc_meta = meta.copy()
            unc_meta.update({"count": 1})
            
            with rasterio.open(unc_path, "w", **unc_meta) as dst:
                dst.write(uncertainty.astype(np.float32), 1)
                dst.set_band_description(1, "Uncertainty (Std Dev)")
        
        logger.info(f"Saved: {out_path.name}")
        return out_path

    def _find_file_for_year(self, year: int) -> Path:
        """Locate feature file with flexible naming convention."""
        for pattern in [f"{year}_features_complete.tif", f"{year}_features.tif"]:
            matches = list(self.data_dir.glob(pattern))
            if matches:
                return matches[0]
        raise FileNotFoundError(f"No features found for year {year} in {self.data_dir}")

    def _infer_interval(self) -> int:
        """Determine temporal resolution of the dataset."""
        files = sorted(self.data_dir.glob("*_features*.tif"))
        if not files:
            raise FileNotFoundError("No feature files found to infer interval.")
            
        years = sorted(list(set(self._extract_year(f.name) for f in files)))

        if len(years) < 2:
            return 2  # Default assumption if only 1 file exists

        intervals = [years[i+1] - years[i] for i in range(len(years)-1)]
        return min(intervals)

    def _calculate_prediction_stats(self, prediction: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute basic descriptive statistics for the prediction."""
        stats = {}
        for i, name in enumerate(self.band_names):
            band_data = prediction[i]
            stats[name] = {
                "min": float(np.min(band_data)),
                "max": float(np.max(band_data)),
                "mean": float(np.mean(band_data)),
                "std": float(np.std(band_data)),
                "median": float(np.median(band_data)),
            }
        return stats

    def _get_raster_metadata(self, year: int) -> Dict:
        """Retrieve metadata from the source file of a specific year."""
        path = self._find_file_for_year(year)
        with rasterio.open(path) as src:
            return src.meta.copy()

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract 4-digit year from filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")
