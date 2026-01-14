"""
Band Processing and Spectral Indices Calculation

Computes NDBI, NDVI, NDWI, NDBSI, and LST from Landsat bands.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


class BandProcessor:
    """
    Process Landsat bands and calculate spectral indices.

    Handles Landsat 5/7/8/9 with appropriate band mappings.

    Args:
        landsat_version: Landsat satellite version (5, 7, 8, or 9)
    """

    # Band mappings for different Landsat versions
    BAND_MAPPING = {
        5: {  # Landsat 5 TM
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir": "SR_B4",
            "swir1": "SR_B5",
            "thermal": "ST_B6",
            "swir2": "SR_B7",
        },
        7: {  # Landsat 7 ETM+
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir": "SR_B4",
            "swir1": "SR_B5",
            "thermal": "ST_B6",
            "swir2": "SR_B7",
        },
        8: {  # Landsat 8 OLI/TIRS
            "coastal": "SR_B1",
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
            "thermal": "ST_B10",
        },
        9: {  # Landsat 9 OLI-2/TIRS-2
            "coastal": "SR_B1",
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
            "thermal": "ST_B10",
        },
    }

    def __init__(self, landsat_version: int = 8) -> None:
        if landsat_version not in self.BAND_MAPPING:
            raise ValueError(f"Unsupported Landsat version: {landsat_version}")

        self.landsat_version = landsat_version
        self.band_map = self.BAND_MAPPING[landsat_version]

        logger.info(f"BandProcessor initialized for Landsat {landsat_version}")

    def process_raster(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        calculate_all: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Process Landsat raster and calculate all indices.

        Args:
            input_path: Path to input GeoTIFF
            output_path: Path to save processed output (optional)
            calculate_all: Calculate all indices

        Returns:
            Dictionary of calculated indices
        """
        logger.info(f"Processing: {input_path.name}")

        # Read bands
        with rasterio.open(input_path) as src:
            metadata = src.meta.copy()
            bands = self._read_bands(src)

        # Calculate indices
        results = {}

        if calculate_all:
            results["NDBI"] = self.calculate_ndbi(bands["red"], bands["swir1"], bands["nir"])
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
        """Read required bands from raster with auto-detection."""
        bands = {}
        descriptions = list(src.descriptions or [])
        
        if not descriptions:
            raise ValueError("Raster has no band descriptions")
        
        # Auto-detect Landsat version from thermal band
        if "ST_B6" in descriptions:
            # Landsat 5/7
            logger.debug("Detected Landsat 5/7 from ST_B6 thermal band")
            band_map = self.BAND_MAPPING[5]
        elif "ST_B10" in descriptions:
            # Landsat 8/9
            logger.debug("Detected Landsat 8/9 from ST_B10 thermal band")
            band_map = self.BAND_MAPPING[8]
        else:
            # Fallback to initialized version
            logger.warning(f"Could not auto-detect Landsat version. Using configured: L{self.landsat_version}")
            band_map = self.band_map
        
        # Read bands using detected mapping
        for common_name, band_name in band_map.items():
            if band_name in descriptions:
                band_idx = descriptions.index(band_name) + 1
                bands[common_name] = src.read(band_idx).astype(np.float32)
        
        # Validate required bands are present
        required = ["red", "nir", "swir1", "thermal", "blue", "green"]
        missing = [b for b in required if b not in bands]
        if missing:
            raise ValueError(
                f"Missing required bands: {missing}\n"
                f"Available bands in file: {descriptions}\n"
                f"Looking for: {list(band_map.values())}"
            )
        
        return bands

    @staticmethod
    def calculate_ndbi(
        red: np.ndarray,
        swir1: np.ndarray,
        nir: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate Normalized Difference Built-up Index.

        NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)

        High values indicate built-up areas.
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

        High values indicate dense vegetation.
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

        High values indicate water bodies.
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

        Combination of bare soil and built-up characteristics.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = (swir1 + red) - (nir + blue)
            denominator = (swir1 + red) + (nir + blue)
            ndbsi = numerator / denominator
            ndbsi = np.where(np.isfinite(ndbsi), ndbsi, 0)
        return ndbsi

    @staticmethod
    def calculate_lst(thermal: np.ndarray, kelvin_to_celsius: bool = True) -> np.ndarray:
        """
        Calculate Land Surface Temperature.

        Assumes thermal band is in Kelvin (after scale factor application).

        Args:
            thermal: Thermal band in Kelvin
            kelvin_to_celsius: Convert to Celsius

        Returns:
            LST in Celsius or Kelvin
        """
        lst = thermal.copy()

        if kelvin_to_celsius:
            lst = lst - 273.15

        return lst

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

        logger.info(f"Saved indices: {output_path}")


class MultiTemporalProcessor:
    """
    Process multiple temporal rasters in sequence.

    Handles batch processing of time series data.
    """

    def __init__(self, landsat_version: int = 8) -> None:
        self.processor = BandProcessor(landsat_version=landsat_version)

    def process_time_series(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.tif",
    ) -> Dict[int, Path]:
        """
        Process all rasters in directory.

        Args:
            input_dir: Directory with input GeoTIFFs
            output_dir: Directory for processed outputs
            pattern: Filename pattern for inputs

        Returns:
            Dictionary mapping year to output path
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
        import re

        # Match YYYY in filename (e.g., L8_GeoTIFF_2023-07-01_2023-12-31_cropped.tif)
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from filename: {filename}")
