"""
Raster Loading and Validation
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class RasterLoader:
    """
    Load and validate raster files.

    Handles Landsat GeoTIFF files with validation.
    """

    def __init__(self, validate: bool = True) -> None:
        self.validate = validate

    def load(
        self,
        path: Path,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Load raster file.

        Args:
            path: Path to GeoTIFF
            bands: Specific bands to load (None = all)

        Returns:
            Tuple of (data array, metadata dict)
        """
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            if bands is None:
                data = src.read()
            else:
                data = src.read(bands)

            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": src.dtypes[0],
                "descriptions": src.descriptions,
            }

        # Validation
        if self.validate:
            self._validate_data(data, path)

        return data, metadata

    def _validate_data(self, data: np.ndarray, path: Path) -> None:
        """Validate loaded data."""
        # Check for NaN values
        if np.isnan(data).any():
            logger.warning(f"NaN values detected in {path.name}")

        # Check for infinite values
        if np.isinf(data).any():
            logger.warning(f"Infinite values detected in {path.name}")

        # Check value ranges
        if data.min() < -1e6 or data.max() > 1e6:
            logger.warning(f"Unusual value range in {path.name}: [{data.min()}, {data.max()}]")
