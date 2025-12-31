"""File Readers"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import yaml

logger = logging.getLogger(__name__)


class RasterReader:
    """Read raster files with validation."""

    @staticmethod
    def read(
        path: Path,
        bands: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Read raster file.

        Args:
            path: Path to raster
            bands: Specific bands to read (None = all)

        Returns:
            Tuple of (data, metadata)
        """
        if not path.exists():
            raise FileNotFoundError(f"Raster not found: {path}")

        with rasterio.open(path) as src:
            # Read data
            if bands is None:
                data = src.read()
            else:
                data = src.read(bands)

            # Extract metadata
            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "descriptions": list(src.descriptions) if src.descriptions else None,
                "nodata": src.nodata,
            }

        logger.debug(f"Read raster: {path.name} ({data.shape})")
        return data, metadata

    @staticmethod
    def read_band(path: Path, band: int) -> np.ndarray:
        """Read single band."""
        with rasterio.open(path) as src:
            return src.read(band)

    @staticmethod
    def get_metadata(path: Path) -> Dict:
        """Get metadata without reading data."""
        _, metadata = RasterReader.read(path)
        return metadata


class CSVReader:
    """Read CSV files."""

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        """Read CSV file."""
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)
        logger.debug(f"Read CSV: {path.name} ({len(df)} rows)")
        return df


class ConfigReader:
    """Read configuration files."""

    @staticmethod
    def read_yaml(path: Path) -> Dict:
        """Read YAML configuration."""
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        logger.debug(f"Read config: {path.name}")
        return config

    @staticmethod
    def read_json(path: Path) -> Dict:
        """Read JSON configuration."""
        import json

        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path) as f:
            config = json.load(f)

        logger.debug(f"Read config: {path.name}")
        return config
