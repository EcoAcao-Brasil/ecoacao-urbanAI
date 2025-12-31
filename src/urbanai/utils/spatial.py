"""Spatial Utilities"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from typing import Tuple
from pathlib import Path

def reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs: str = "EPSG:4326",
    resampling: Resampling = Resampling.bilinear,
) -> None:
    """
    Reproject raster to different CRS.

    Args:
        src_path: Source raster path
        dst_path: Destination raster path
        dst_crs: Target CRS
        resampling: Resampling method
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling,
                )


def calculate_pixel_area(
    transform: rasterio.Affine,
    crs: str,
) -> float:
    """
    Calculate area of a single pixel in square meters.

    Args:
        transform: Raster transform
        crs: Coordinate reference system

    Returns:
        Pixel area in m²
    """
    # Get pixel dimensions
    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    # If in geographic coordinates, approximate
    if "4326" in str(crs):
        # Rough approximation at equator: 1 degree ≈ 111 km
        lat_meters = pixel_height * 111000
        lon_meters = pixel_width * 111000
        return lat_meters * lon_meters

    # For projected CRS, units are typically meters
    return pixel_width * pixel_height


def extract_patch(
    raster: np.ndarray,
    center: Tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    """
    Extract square patch from raster.

    Args:
        raster: Input raster (channels, height, width)
        center: Center coordinates (y, x)
        patch_size: Size of patch

    Returns:
        Extracted patch
    """
    cy, cx = center
    half_size = patch_size // 2

    y_start = max(0, cy - half_size)
    y_end = min(raster.shape[1], cy + half_size)
    x_start = max(0, cx - half_size)
    x_end = min(raster.shape[2], cx + half_size)

    patch = raster[:, y_start:y_end, x_start:x_end]

    # Pad if necessary
    if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
        padded = np.zeros((raster.shape[0], patch_size, patch_size), dtype=raster.dtype)
        padded[:, :patch.shape[1], :patch.shape[2]] = patch
        return padded
    return patch
