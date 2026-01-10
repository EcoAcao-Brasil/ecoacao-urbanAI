"""
Residual Analysis

Calculate temporal changes between current and future predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class ResidualCalculator:
    """
    Calculate residuals between current and future predictions.

    Residuals = Future - Current (shows the change expected over time)

    Args:
        current_raster: Path to current year raster
        future_raster: Path to predicted future raster
        output_dir: Directory for outputs
        weights: Optional dictionary of band weights for combined residual.
                 If not provided, uses defaults: LST=0.4, IS=0.3, SS=0.2, NDBI=0.1
    """

    def __init__(
        self,
        current_raster: Path,
        future_raster: Path,
        output_dir: Optional[Path] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.current_raster = Path(current_raster)
        self.future_raster = Path(future_raster)
        self.output_dir = Path(output_dir) if output_dir else Path("residuals")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set weights with defaults
        self.weights = weights or {
            "LST": 0.4,
            "IS": 0.3,
            "SS": 0.2,
            "NDBI": 0.1,
        }
        
        # Normalize weights to sum to 1.0
        self._normalize_weights()

        logger.info("ResidualCalculator initialized")
        logger.info(f"Current: {self.current_raster.name}")
        logger.info(f"Future: {self.future_raster.name}")
        logger.info(f"Using priority weights: {self.weights}")

    def _normalize_weights(self) -> None:
        """
        Normalize weights to sum to 1.0.
        
        This ensures that the combined residual is properly scaled
        regardless of the input weights.
        """
        total = sum(self.weights.values())
        if total == 0:
            logger.warning("All weights are zero, using equal weights")
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights.keys()}
        elif abs(total - 1.0) > 1e-6:
            logger.info(f"Normalizing weights (sum={total:.3f}) to sum to 1.0")
            self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate_all_residuals(self) -> Dict[str, Path]:
        """
        Calculate residuals for all bands.

        Returns:
            Dictionary mapping metric names to residual file paths
        """
        logger.info("Calculating residuals...")

        # Load rasters
        with rasterio.open(self.current_raster) as src_current:
            current_data = src_current.read()
            metadata = src_current.meta.copy()
            descriptions = src_current.descriptions

        with rasterio.open(self.future_raster) as src_future:
            future_data = src_future.read()

        if current_data.shape != future_data.shape:
            raise ValueError(
                f"Raster shapes don't match: "
                f"current {current_data.shape} vs future {future_data.shape}"
            )

        # Calculate residuals (future - current)
        residuals = future_data - current_data

        # Save per-band residuals
        output_paths = {}
        band_names = descriptions or [f"Band_{i+1}" for i in range(current_data.shape[0])]

        for i, name in enumerate(band_names):
            output_path = self._save_residual(
                residuals[i],
                metadata,
                name,
            )
            output_paths[name] = output_path
            logger.info(f"  {name}: range [{residuals[i].min():.3f}, {residuals[i].max():.3f}]")

        # Calculate combined residual (weighted by importance)
        combined = self._calculate_combined_residual(residuals, band_names)
        output_paths["combined_residuals"] = self._save_residual(
            combined,
            metadata,
            "combined",
        )

        logger.info(f"Calculated {len(output_paths)} residual maps")
        return output_paths

    def calculate_band_residual(self, band_name: str) -> np.ndarray:
        """
        Calculate residual for specific band.

        Args:
            band_name: Name of band to calculate

        Returns:
            Residual array
        """
        with rasterio.open(self.current_raster) as src:
            descriptions = list(src.descriptions or [])
            if band_name not in descriptions:
                raise ValueError(f"Band {band_name} not found")
            
            band_idx = descriptions.index(band_name) + 1
            current = src.read(band_idx)

        with rasterio.open(self.future_raster) as src:
            future = src.read(band_idx)

        return future - current

    def _save_residual(
        self,
        residual: np.ndarray,
        metadata: Dict,
        name: str,
    ) -> Path:
        """Save residual to GeoTIFF."""
        output_path = self.output_dir / f"{name}_residual.tif"

        meta = metadata.copy()
        meta.update({"count": 1, "dtype": "float32"})

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(residual.astype(np.float32), 1)
            dst.set_band_description(1, f"{name}_residual")

        logger.debug(f"Saved residual: {output_path}")
        return output_path

    def _calculate_combined_residual(
        self,
        residuals: np.ndarray,
        band_names: List[str],
    ) -> np.ndarray:
        """
        Calculate combined residual with weighted importance.

        Priority weights: LST > IS > SS > NDBI > others
        
        Args:
            residuals: Array of residuals (bands, h, w)
            band_names: List of band names

        Returns:
            Combined residual array (h, w)
        """
        combined = np.zeros_like(residuals[0])

        for i, name in enumerate(band_names):
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                # Normalize to [-1, 1] range before combining
                normalized = self._normalize_residual(residuals[i])
                combined += weight * normalized
            elif name in self.weights:
                # Log if weight is zero but band is in weights dict
                logger.debug(f"Skipping band {name} with zero weight")

        # Warn about unknown bands that have data but no weights
        unknown_bands = [name for name in band_names if name not in self.weights and name]
        if unknown_bands:
            logger.warning(f"Unknown bands in data (not weighted): {unknown_bands}")

        logger.info(f"Combined residual range: [{combined.min():.3f}, {combined.max():.3f}]")
        return combined

    @staticmethod
    def _normalize_residual(residual: np.ndarray) -> np.ndarray:
        """
        Normalize residual to [-1, 1] range.
        
        Args:
            residual: Input residual array

        Returns:
            Normalized residual
        """
        abs_max = np.abs(residual).max()
        if abs_max > 0:
            return residual / abs_max
        return residual

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all residuals.

        Returns:
            Dictionary of statistics per band
        """
        with rasterio.open(self.current_raster) as src_current:
            current_data = src_current.read()
            descriptions = src_current.descriptions or []

        with rasterio.open(self.future_raster) as src_future:
            future_data = src_future.read()

        residuals = future_data - current_data

        stats = {}
        for i, name in enumerate(descriptions):
            band_residual = residuals[i]
            stats[name] = {
                "min": float(np.min(band_residual)),
                "max": float(np.max(band_residual)),
                "mean": float(np.mean(band_residual)),
                "std": float(np.std(band_residual)),
                "median": float(np.median(band_residual)),
                "n_positive": int(np.sum(band_residual > 0)),
                "n_negative": int(np.sum(band_residual < 0)),
            }

        return stats
      
