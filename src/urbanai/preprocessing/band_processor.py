"""
Band Processing and Spectral Indices Calculation

Computes NDBI, NDVI, NDWI, NDBSI, and LST from Landsat bands.

Band reading by description name + resume capability.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class BandProcessor:
    """
    Process Landsat bands and calculate spectral indices.

    Handles Landsat 5/7/8/9 with automatic band detection by name.
    """

    def __init__(self, landsat_version: int = 8) -> None:
        self.landsat_version = landsat_version
        logger.info(f"BandProcessor initialized (auto-detection enabled)")

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

        # Read bands BY DESCRIPTION NAME (not position)
        with rasterio.open(input_path) as src:
            metadata = src.meta.copy()
            bands = self._read_bands_by_name(src)

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

    def _read_bands_by_name(self, src: rasterio.DatasetReader) -> Dict[str, np.ndarray]:
        """
        Read bands by DESCRIPTION name, not position.
        
        This is critical because GEE exports have QA bands interspersed,
        so band positions vary between Landsat versions and exports.
        """
        descriptions = list(src.descriptions or [])
        
        if not descriptions:
            raise ValueError("Raster has no band descriptions")
        
        logger.debug(f"Available bands: {descriptions}")
        
        bands = {}
        
        # Define flexible band mappings (handles both L5/7 and L8/9)
        band_mappings = {
            "blue": ["SR_B1", "SR_B2"],
            "green": ["SR_B2", "SR_B3"],
            "red": ["SR_B3", "SR_B4"],
            "nir": ["SR_B4", "SR_B5"],
            "swir1": ["SR_B5", "SR_B6", "SR_B7"],
            "swir2": ["SR_B7"],
            "thermal": ["ST_B6", "ST_B10"],
        }
        
        # Detect Landsat version by checking which bands exist
        if "SR_B6" in descriptions and "ST_B10" in descriptions:
            # Landsat 8/9 format
            band_mappings = {
                "blue": ["SR_B2"],
                "green": ["SR_B3"],
                "red": ["SR_B4"],
                "nir": ["SR_B5"],
                "swir1": ["SR_B6"],
                "swir2": ["SR_B7"],
                "thermal": ["ST_B10"],
            }
            logger.debug("Detected Landsat 8/9 format")
        else:
            # Landsat 5/7 format
            band_mappings = {
                "blue": ["SR_B1"],
                "green": ["SR_B2"],
                "red": ["SR_B3"],
                "nir": ["SR_B4"],
                "swir1": ["SR_B5"],
                "swir2": ["SR_B7"],
                "thermal": ["ST_B6"],
            }
            logger.debug("Detected Landsat 5/7 format")
        
        # Read each required band by searching for its name
        for common_name, possible_names in band_mappings.items():
            found = False
            for band_name in possible_names:
                if band_name in descriptions:
                    band_idx = descriptions.index(band_name) + 1
                    bands[common_name] = src.read(band_idx).astype(np.float32)
                    logger.debug(f"  {common_name} → {band_name} (band {band_idx})")
                    found = True
                    break
            
            if not found:
                logger.warning(f"Could not find band for {common_name}")
        
        # Validate required bands are present
        required = ["red", "nir", "swir1", "thermal", "blue", "green"]
        missing = [b for b in required if b not in bands]
        if missing:
            raise ValueError(
                f"Missing required bands: {missing}\n"
                f"Available bands: {descriptions}\n"
                f"Band mappings used: {band_mappings}"
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
    Process multiple temporal rasters in sequence with resume capability.

    Handles batch processing of time series data.
    """

    def __init__(self, landsat_version: int = 8) -> None:
        self.processor = BandProcessor(landsat_version=landsat_version)

    def process_time_series(
        self,
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*.tif",
        resume: bool = True,
    ) -> Dict[int, Path]:
        """
        Process all rasters in directory with resume support.

        Args:
            input_dir: Directory with input GeoTIFFs
            output_dir: Directory for processed outputs
            pattern: Filename pattern for inputs
            resume: Skip already-processed years

        Returns:
            Dictionary mapping year to output path
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(input_dir.glob(pattern))
        if not input_files:
            raise ValueError(f"No files matching {pattern} in {input_dir}")

        # Create progress tracker
        progress_file = output_dir / ".processing_progress.json"
        completed_years = self._load_progress(progress_file) if resume else set()

        logger.info(f"Processing {len(input_files)} rasters")
        if completed_years:
            logger.info(f"Resuming: {len(completed_years)} years already processed")

        results = {}
        for input_path in input_files:
            # Extract year from filename
            year = self._extract_year(input_path.name)
            output_path = output_dir / f"{year}_features.tif"

            # Skip if already processed
            if resume and year in completed_years and output_path.exists():
                logger.info(f"✓ Skipping {year} (already processed)")
                results[year] = output_path
                continue

            # Process this year
            logger.info(f"Processing {year}...")
            
            try:
                self.processor.process_raster(
                    input_path=input_path,
                    output_path=output_path,
                    calculate_all=True,
                )

                results[year] = output_path
                
                # Mark as completed
                completed_years.add(year)
                self._save_progress(progress_file, completed_years)
                
            except Exception as e:
                logger.error(f"Failed to process {year}: {e}")
                continue

        logger.info(f"Processed {len(results)} temporal rasters")
        return results

    def _load_progress(self, progress_file: Path) -> set:
        """Load completed years from progress file."""
        if not progress_file.exists():
            return set()
        
        try:
            with open(progress_file) as f:
                data = json.load(f)
                return set(data.get("completed_years", []))
        except:
            return set()

    def _save_progress(self, progress_file: Path, completed_years: set) -> None:
        """Save completed years to progress file."""
        with open(progress_file, "w") as f:
            json.dump({"completed_years": sorted(list(completed_years))}, f, indent=2)

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re

        # Match YYYY in filename (e.g., L8_GeoTIFF_2023-07-01_2023-12-31_cropped.tif)
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from filename: {filename}")
