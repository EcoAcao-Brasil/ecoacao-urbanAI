"""
Future Prediction Engine

Generates future urban heat predictions using trained models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import rasterio
import torch
from tqdm import tqdm

from urbanai.models import ConvLSTMEncoderDecoder

logger = logging.getLogger(__name__)


class FuturePredictor:
    """
    Predict future urban heat landscapes.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Directory with processed features
        output_dir: Directory for predictions
        device: Compute device
    """

    def __init__(
        self,
        model_path: Path,
        data_dir: Path,
        output_dir: Optional[Path] = None,
        device: str = "cuda",
    ) -> None:
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "predictions"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load model
        self.model, self.config = self._load_model()
        self.model.eval()

        logger.info("FuturePredictor initialized")
        logger.info(f"Model: {model_path.name}")
        logger.info(f"Device: {self.device}")

    def predict(
        self,
        current_year: int,
        target_year: int,
        save_outputs: bool = True,
        calculate_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict future urban heat landscape.

        Args:
            current_year: Most recent year in training data
            target_year: Year to predict
            save_outputs: Save prediction rasters
            calculate_uncertainty: Calculate prediction uncertainty

        Returns:
            Dictionary with predictions and metadata
        """
        logger.info(f"Predicting {target_year} from {current_year}")

        # Calculate number of steps
        year_interval = self._infer_interval()
        n_steps = (target_year - current_year) // year_interval

        if n_steps <= 0:
            raise ValueError(f"Target year {target_year} must be after {current_year}")

        logger.info(f"Prediction steps: {n_steps} (interval: {year_interval} years)")

        # Load input sequence
        input_sequence = self._load_input_sequence(current_year)

        # Get metadata from last file
        last_file = self._get_file_for_year(current_year)
        with rasterio.open(last_file) as src:
            metadata = src.meta.copy()

        # Make prediction
        with torch.no_grad():
            prediction = self._predict_multi_step(input_sequence, n_steps)

        # Denormalize if needed
        if self.config.get("normalize", True):
            prediction = self._denormalize(prediction)

        # Calculate uncertainty if requested
        uncertainty = None
        if calculate_uncertainty:
            uncertainty = self._calculate_uncertainty(input_sequence, n_steps)

        # Save outputs
        output_path = None
        if save_outputs:
            output_path = self._save_prediction(
                prediction,
                metadata,
                target_year,
                uncertainty,
            )

        # Calculate statistics
        stats = self._calculate_prediction_stats(prediction)

        results = {
            "target_year": target_year,
            "prediction_shape": prediction.shape,
            "output_path": output_path,
            "statistics": stats,
        }

        if uncertainty is not None:
            results["uncertainty"] = uncertainty

        logger.info(f"Prediction complete: {target_year}")
        return results

    def _load_model(self) -> tuple[torch.nn.Module, Dict]:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract config
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})

        # Build model
        model = ConvLSTMEncoderDecoder(
            input_channels=model_config.get("input_channels", 7),
            hidden_dims=model_config.get("hidden_dims", [64, 128, 256, 256, 128, 64]),
            kernel_size=(model_config.get("kernel_size", 3), model_config.get("kernel_size", 3)),
            num_encoder_layers=model_config.get("num_encoder_layers", 3),
            num_decoder_layers=model_config.get("num_decoder_layers", 3),
            output_channels=model_config.get("output_channels", 7),
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        logger.info("Model loaded successfully")
        return model, config

    def _load_input_sequence(self, current_year: int) -> torch.Tensor:
        """Load input sequence for prediction."""
        sequence_length = self.config.get("sequence_length", 10)
        year_interval = self._infer_interval()

        # Calculate years in sequence
        years = [
            current_year - i * year_interval
            for i in range(sequence_length - 1, -1, -1)
        ]

        logger.info(f"Loading sequence: {years[0]}-{years[-1]}")

        # Load rasters
        sequence_data = []
        for year in years:
            file_path = self._get_file_for_year(year)
            with rasterio.open(file_path) as src:
                data = src.read()  # (channels, h, w)
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                sequence_data.append(data)

        # Stack into sequence
        sequence = np.stack(sequence_data, axis=0)  # (seq_len, channels, h, w)

        # Normalize if needed
        if self.config.get("normalize", True):
            sequence = self._normalize(sequence)

        # Convert to tensor
        tensor = torch.from_numpy(sequence).float().unsqueeze(0)  # (1, seq_len, ch, h, w)
        tensor = tensor.to(self.device)

        return tensor

    def _predict_multi_step(
        self,
        input_sequence: torch.Tensor,
        n_steps: int,
    ) -> np.ndarray:
        """
        Predict multiple steps into the future.

        Args:
            input_sequence: Input tensor (1, seq_len, channels, h, w)
            n_steps: Number of steps to predict

        Returns:
            Predicted array (channels, h, w) for final step
        """
        current_sequence = input_sequence

        logger.info(f"Predicting {n_steps} steps...")

        for step in tqdm(range(n_steps), desc="Prediction steps"):
            # Predict next step
            with torch.no_grad():
                prediction = self.model(current_sequence, future_steps=1)

            # Update sequence (remove oldest, add prediction)
            # prediction: (1, 1, channels, h, w)
            # current_sequence: (1, seq_len, channels, h, w)
            current_sequence = torch.cat([
                current_sequence[:, 1:, :, :, :],  # Remove first timestep
                prediction,  # Add prediction
            ], dim=1)

        # Get final prediction
        final_prediction = prediction[0, 0].cpu().numpy()  # (channels, h, w)

        return final_prediction

    def _calculate_uncertainty(
        self,
        input_sequence: torch.Tensor,
        n_steps: int,
        n_samples: int = 10,
    ) -> np.ndarray:
        """
        Calculate prediction uncertainty using Monte Carlo dropout.

        Args:
            input_sequence: Input sequence
            n_steps: Number of prediction steps
            n_samples: Number of MC samples

        Returns:
            Uncertainty map (h, w)
        """
        logger.info("Calculating uncertainty...")

        # Enable dropout for MC sampling
        self.model.train()

        predictions = []
        for _ in tqdm(range(n_samples), desc="MC samples"):
            pred = self._predict_multi_step(input_sequence.clone(), n_steps)
            predictions.append(pred)

        predictions = np.stack(predictions, axis=0)  # (n_samples, channels, h, w)

        # Calculate uncertainty as standard deviation
        uncertainty = np.std(predictions, axis=0)  # (channels, h, w)

        # Average across channels
        uncertainty_map = np.mean(uncertainty, axis=0)  # (h, w)

        self.model.eval()

        return uncertainty_map

    def _save_prediction(
        self,
        prediction: np.ndarray,
        metadata: Dict,
        year: int,
        uncertainty: Optional[np.ndarray] = None,
    ) -> Path:
        """Save prediction to GeoTIFF."""
        # Update metadata
        meta = metadata.copy()
        meta.update({
            "count": prediction.shape[0],
            "dtype": "float32",
        })

        # Save prediction
        output_path = self.output_dir / f"{year}_predicted.tif"
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(prediction.astype(np.float32))
            # Set band descriptions
            descriptions = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST", "IS", "SS"]
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)

        logger.info(f"Saved prediction: {output_path}")

        # Save uncertainty if available
        if uncertainty is not None:
            uncertainty_path = self.output_dir / f"{year}_uncertainty.tif"
            meta_unc = meta.copy()
            meta_unc["count"] = 1

            with rasterio.open(uncertainty_path, "w", **meta_unc) as dst:
                dst.write(uncertainty.astype(np.float32), 1)

            logger.info(f"Saved uncertainty: {uncertainty_path}")

        return output_path

    def _get_file_for_year(self, year: int) -> Path:
        """Get file path for specific year."""
        # Try complete features first
        pattern = f"{year}_features_complete.tif"
        files = list(self.data_dir.glob(pattern))

        if not files:
            # Try regular features
            pattern = f"{year}_features.tif"
            files = list(self.data_dir.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No file found for year {year}")

        return files[0]

    def _infer_interval(self) -> int:
        """Infer year interval from available files."""
        files = sorted(self.data_dir.glob("*_features*.tif"))
        years = sorted([self._extract_year(f.name) for f in files])

        if len(years) < 2:
            return 2  # Default to biennial

        intervals = [years[i+1] - years[i] for i in range(len(years)-1)]
        return min(intervals)  # Use minimum interval

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize data (placeholder - should use training stats)."""
        # TODO: Load normalization stats from training
        return data

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """Denormalize data (placeholder)."""
        # TODO: Load normalization stats from training
        return data

    def _calculate_prediction_stats(self, prediction: np.ndarray) -> Dict:
        """Calculate prediction statistics."""
        band_names = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST", "IS", "SS"]

        stats = {}
        for i, name in enumerate(band_names):
            band_data = prediction[i]
            stats[name] = {
                "min": float(np.min(band_data)),
                "max": float(np.max(band_data)),
                "mean": float(np.mean(band_data)),
                "std": float(np.std(band_data)),
            }

        return stats

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re

        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")
