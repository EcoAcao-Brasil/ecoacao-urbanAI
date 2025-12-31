"""Data Validation Utilities"""

from pathlib import Path
from typing import List
import rasterio


def validate_raster_files(
    file_paths: List[Path],
    required_bands: int = 7,
    check_crs: bool = True,
    check_shape: bool = True,
) -> bool:
    """
    Validate raster files for consistency.

    Args:
        file_paths: List of raster file paths
        required_bands: Expected number of bands
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
            f"Expected {required_bands} bands, found {ref_bands} in {file_paths[0]}"
        )

    # Check remaining files
    for path in file_paths[1:]:
        with rasterio.open(path) as src:
            if check_crs and src.crs != ref_crs:
                raise ValueError(f"CRS mismatch in {path}")

            if check_shape and (src.height, src.width) != ref_shape:
                raise ValueError(f"Shape mismatch in {path}")

            if src.count != required_bands:
                raise ValueError(f"Band count mismatch in {path}")

    return True
