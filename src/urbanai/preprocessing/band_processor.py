"""
Band Processing and Spectral Indices Calculation

Computes NDBI, NDVI, NDWI, NDBSI, and LST from Pre-scaled Landsat bands.
Assumes inputs are from GEE export with applyScaleFactors() already run.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
import re

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class BandProcessor:
    """
    Process Landsat bands with explicit user-defined band mapping.

    This class assumes data is ALREADY SCALED (Reflectance 0-1, Kelvin).
    It does not apply USGS scale factors.

    Args:
        band_mapping: Dictionary mapping common band names to actual band descriptions.
            Required keys: 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'thermal'

    Raises:
        ValueError: If band_mapping is missing required keys.
    """

    REQUIRED_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]

    def __init__(self, band_mapping: Dict[str, str]) -> None:
        missing = [b for b in self.REQUIRED_BANDS if b not in band_mapping]

        if missing:
            raise ValueError(
                f"Missing required bands in mapping: {missing}. "
                f"Required bands: {self.REQUIRED_BANDS}"
            )

        self.band_mapping = band_mapping
        logger.info("BandProcessor initialized with user-defined mapping")
        for common, actual in band_mapping.items():
            logger.debug(f"  {common}: {actual}")

    def process_raster(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        calculate_all: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Process Landsat raster and calculate spectral indices.

        Args:
            input_path: Path to input GeoTIFF file.
            output_path: Path to save processed indices. If None, no file is saved.
            calculate_all: Whether to calculate all indices.

        Returns:
            Dictionary with keys 'NDBI', 'NDVI', 'NDWI', 'NDBSI', 'LST'
            mapping to numpy arrays.
        """
        logger.info(f"Processing raster: {input_path.name}")

        # Read bands
        with rasterio.open(input_path) as src:
            metadata = src.meta.copy()
            bands = self._read_bands(src)

        # Calculate indices
        results = {}

        if calculate_all:
            results["NDBI"] = self.calculate_ndbi(
                bands["red"], bands["swir1"], bands["nir"]
            )
            results["NDVI"] = self.calculate_ndvi(bands["red"], bands["nir"])
            results["NDWI"] = self.calculate_ndwi(bands["green"], bands["nir"])
            results["NDBSI"] = self.calculate_ndbsi(
                bands["blue"], bands["red"], bands["nir"], bands["swir1"]
            )
            results["LST"] = self.calculate_lst(bands["thermal"])

        # Save if output path provided
        if output_path:
            self._save_indices(results, output_path, metadata)

        logger.info(f"Calculated {len(results)} indices")
        return results

    def _read_bands(self, src: rasterio.DatasetReader) -> Dict[str, np.ndarray]:
        """
        Read bands from raster using user-defined mapping.
        """
        descriptions = list(src.descriptions or [])

        if not descriptions:
            raise ValueError("Raster contains no band descriptions")

        logger.debug(f"Available bands in raster: {descriptions}")

        bands = {}

        for common_name, band_name in self.band_mapping.items():
            if band_name not in descriptions:
                # Fallback: check if band_name exists without the description mapping
                # strictly speaking, we rely on descriptions here.
                raise ValueError(
                    f"Band '{band_name}' (mapped to '{common_name}') not found. "
                    f"Available bands: {descriptions}"
                )

            band_idx = descriptions.index(band_name) + 1
            bands[common_name] = src.read(band_idx).astype(np.float32)

            logger.debug(f"Read {common_name}: {band_name} at index {band_idx}")

        self._validate_thermal(bands["thermal"], self.band_mapping["thermal"])

        return bands

    def _validate_thermal(self, thermal_data: np.ndarray, thermal_name: str) -> None:
        """
        Validate thermal band data ranges (Expected: Kelvin).
        """
        thermal_min = np.nanmin(thermal_data)
        thermal_max = np.nanmax(thermal_data)
        thermal_mean = np.nanmean(thermal_data)

        logger.debug(f"Thermal band ({thermal_name}) statistics (Kelvin):")
        logger.debug(f"  Range: [{thermal_min:.2f}, {thermal_max:.2f}]")
        logger.debug(f"  Mean: {thermal_mean:.2f}")

        # Validation for Kelvin ranges (approx 200K - 350K)
        if thermal_max > 1000:
            logger.warning(
                f"Thermal band maximum ({thermal_max:.2f}) is suspiciously high. "
                "Expected Kelvin (approx 300-330). "
                "Did you forget to apply scale factors in GEE?"
            )
        elif thermal_max < 200 and thermal_max > 0:
            logger.warning(
                f"Thermal band maximum ({thermal_max:.2f}) is suspiciously low for Kelvin. "
                "Check input data units."
            )

    @staticmethod
    def calculate_ndbi(
        red: np.ndarray,
        swir1: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Built-up Index.
        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ndbi = (swir1 - nir) / (swir1 + nir)
            ndbi = np.where(np.isfinite(ndbi), ndbi, 0)
        return ndbi

    @staticmethod
    def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index.
        NDVI = (NIR - RED) / (NIR + RED)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.where(np.isfinite(ndvi), ndvi, 0)
        return ndvi

    @staticmethod
    def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Calculate Normalized Difference Water Index.
        NDWI = (GREEN - NIR) / (GREEN + NIR)
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = (green - nir) / (green + nir)
            ndwi = np.where(np.isfinite(ndwi), ndwi, 0)
        return ndwi

    @staticmethod
    def calculate_ndbsi(
        blue: np.ndarray,
        red: np.ndarray,
        nir: np.ndarray,
        swir1: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Bareness and Soil Index.
        NDBSI = ((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = (swir1 + red) - (nir + blue)
            denominator = (swir1 + red) + (nir + blue)
            ndbsi = numerator / denominator
            ndbsi = np.where(np.isfinite(ndbsi), ndbsi, 0)
        return ndbsi

    @staticmethod
    def calculate_lst(
        thermal: np.ndarray, kelvin_to_celsius: bool = True
    ) -> np.ndarray:
        """
        Calculate Land Surface Temperature.

        Assumes input 'thermal' is already in Kelvin (from GEE export).
        Simply subtracts 273.15 to get Celsius.
        """
        if kelvin_to_celsius:
            # Simple conversion: K -> C
            return thermal - 273.15
        return thermal

    def _save_indices(
        self,
        indices: Dict[str, np.ndarray],
        output_path: Path,
        base_metadata: Dict,
    ) -> None:
        """Save calculated indices to GeoTIFF."""
        # Update metadata
        metadata = base_metadata.copy()
        metadata.update(
            {
                "driver": "GTiff",
                "count": len(indices),
                "dtype": "float32",
                "compress": "lzw",
            }
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write indices
        with rasterio.open(output_path, "w", **metadata) as dst:
            for i, (name, data) in enumerate(indices.items(), start=1):
                dst.write(data.astype(np.float32), i)
                dst.set_band_description(i, name)

        logger.info(f"Saved indices to: {output_path}")


class MultiTemporalProcessor:
    """
    Process multiple temporal rasters in sequence.
    """

    def __init__(self, band_mapping: Dict[str, str]) -> None:
        self.processor = BandProcessor(band_mapping=band_mapping)

    def process_time_series(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.tif",
    ) -> Dict[int, Path]:
        """
        Process all rasters in directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(input_dir.glob(pattern))
        if not input_files:
            raise ValueError(f"No files matching {pattern} in {input_dir}")

        logger.info(f"Processing {len(input_files)} rasters")

        results = {}
        for input_path in input_files:
            # Extract year from filename
            year = self._extract_year(input_path.name)

            # Process
            output_path = output_dir / f"{year}_features.tif"
            self.processor.process_raster(
                input_path=input_path,
                output_path=output_path,
                calculate_all=True,
            )

            results[year] = output_path

        logger.info(f"Processed {len(results)} temporal rasters")
        return results

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        # Match YYYY in filename
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from filename: {filename}")
