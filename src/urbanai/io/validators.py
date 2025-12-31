"""Data Validators"""

import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

import rasterio

logger = logging.getLogger(__name__)


class RasterValidator:
    """Validate raster files."""

    @staticmethod
    def validate_files(
        file_paths: List[Path],
        required_bands: int = 7,
        check_crs: bool = True,
        check_shape: bool = True,
    ) -> bool:
        """
        Validate multiple raster files for consistency.

        Args:
            file_paths: List of raster paths
            required_bands: Expected band count
            check_crs: Check CRS consistency
            check_shape: Check spatial dimensions

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if not file_paths:
            raise ValueError("No files provided")

        # Get reference from first file
        with rasterio.open(file_paths[0]) as ref:
            ref_crs = ref.crs
            ref_shape = (ref.height, ref.width)
            ref_bands = ref.count

        if ref_bands != required_bands:
            raise ValueError(
                f"Expected {required_bands} bands, "
                f"found {ref_bands} in {file_paths[0]}"
            )

        # Check remaining files
        for path in file_paths[1:]:
            with rasterio.open(path) as src:
                if check_crs and src.crs != ref_crs:
                    raise ValueError(f"CRS mismatch in {path}")

                if check_shape and (src.height, src.width) != ref_shape:
                    raise ValueError(
                        f"Shape mismatch in {path}: "
                        f"expected {ref_shape}, got ({src.height}, {src.width})"
                    )

                if src.count != required_bands:
                    raise ValueError(
                        f"Band count mismatch in {path}: "
                        f"expected {required_bands}, got {src.count}"
                    )

        logger.info(f"Validated {len(file_paths)} raster files")
        return True

    @staticmethod
    def check_data_quality(path: Path) -> Dict[str, bool]:
        """
        Check data quality issues.

        Args:
            path: Raster path

        Returns:
            Dictionary of quality checks
        """
        with rasterio.open(path) as src:
            data = src.read()

        checks = {
            "has_nan": bool(np.isnan(data).any()),
            "has_inf": bool(np.isinf(data).any()),
            "has_negative": bool((data < 0).any()),
            "reasonable_range": bool(data.min() > -1e6 and data.max() < 1e6),
        }

        return checks
      
