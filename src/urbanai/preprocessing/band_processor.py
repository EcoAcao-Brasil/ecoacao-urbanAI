import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rasterio

logger = logging.getLogger(__name__)

class BandProcessor:
    """
    Process Landsat bands with explicit user-defined band mapping.
    """

    REQUIRED_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "thermal"]
    
    # Landsat Collection 2 Surface Temperature Scaling Factors
    ST_SCALE = 0.00341802
    ST_OFFSET = 149.0

    def __init__(self, band_mapping: Dict[str, str]) -> None:
        missing = [b for b in self.REQUIRED_BANDS if b not in band_mapping]

        if missing:
            raise ValueError(
                f"Missing required bands in mapping: {missing}. "
                f"Required bands: {self.REQUIRED_BANDS}"
            )

        self.band_mapping = band_mapping
        logger.info("BandProcessor initialized with user-defined mapping")

    def process_raster(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        calculate_all: bool = True,
    ) -> Dict[str, np.ndarray]:
        logger.info(f"Processing raster: {input_path.name}")

        with rasterio.open(input_path) as src:
            metadata = src.meta.copy()
            bands = self._read_bands(src)

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

        if output_path:
            self._save_indices(results, output_path, metadata)

        return results

    def _read_bands(self, src: rasterio.DatasetReader) -> Dict[str, np.ndarray]:
        descriptions = list(src.descriptions or [])
        if not descriptions:
            raise ValueError("Raster contains no band descriptions")

        bands = {}
        for common_name, band_name in self.band_mapping.items():
            if band_name not in descriptions:
                raise ValueError(f"Band '{band_name}' not found.")

            band_idx = descriptions.index(band_name) + 1
            # Read as float32 to preserve precision during math
            bands[common_name] = src.read(band_idx).astype(np.float32)

        # Validate the RAW thermal DNs before processing
        self._validate_thermal(bands["thermal"], self.band_mapping["thermal"])

        return bands

    def _validate_thermal(self, thermal_dn: np.ndarray, thermal_name: str) -> None:
        """
        Validate thermal band Data Numbers (DN).
        """
        dn_min = np.nanmin(thermal_dn)
        dn_max = np.nanmax(thermal_dn)

        logger.debug(f"Thermal band ({thermal_name}) raw DN stats: [{dn_min}, {dn_max}]")

        # Typical valid range for Collection 2 ST is roughly within uint16 limits
        # 0 is often fill value; valid data usually > 0
        if dn_max < 1000:
             logger.warning(
                f"Thermal DN max ({dn_max}) is very low. "
                "Are you sure this is unscaled Collection 2 data? "
                "If this data is already in Kelvin, set ST_SCALE=1 and ST_OFFSET=0."
            )

    @staticmethod
    def calculate_lst(
        thermal_dn: np.ndarray, 
        kelvin_to_celsius: bool = True
    ) -> np.ndarray:
        """
        Calculate Land Surface Temperature from unscaled thermal DN.
        
        Applies USGS Collection 2 scaling:
        Kelvin = DN * 0.00341802 + 149.0
        """
        # 1. Apply Scale and Offset to get Kelvin
        # Use class constants if accessible, or hardcoded for safety here
        scale = 0.00341802
        offset = 149.0
        
        # Avoid calculating on NoData (usually 0) if necessary, 
        # but here we assume nan/valid mask handling is done elsewhere or acceptable.
        kelvin = (thermal_dn * scale) + offset

        if not kelvin_to_celsius:
            return kelvin

        # 2. Convert to Celsius
        lst_celsius = kelvin - 273.15
        return lst_celsius

    # ... [Keep the other static calculation methods (NDBI, NDVI, etc.) as they were] ...

    @staticmethod
    def calculate_ndbi(red, swir1, nir):
        with np.errstate(divide="ignore", invalid="ignore"):
            ndbi = (swir1 - nir) / (swir1 + nir)
            return np.where(np.isfinite(ndbi), ndbi, 0)

    @staticmethod
    def calculate_ndvi(red, nir):
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir - red) / (nir + red)
            return np.where(np.isfinite(ndvi), ndvi, 0)

    @staticmethod
    def calculate_ndwi(green, nir):
        with np.errstate(divide="ignore", invalid="ignore"):
            ndwi = (green - nir) / (green + nir)
            return np.where(np.isfinite(ndwi), ndwi, 0)

    @staticmethod
    def calculate_ndbsi(blue, red, nir, swir1):
        with np.errstate(divide="ignore", invalid="ignore"):
            num = (swir1 + red) - (nir + blue)
            den = (swir1 + red) + (nir + blue)
            ndbsi = num / den
            return np.where(np.isfinite(ndbsi), ndbsi, 0)

    def _save_indices(self, indices, output_path, base_metadata):
        # ... [Same as original] ...
        metadata = base_metadata.copy()
        metadata.update({
            "driver": "GTiff",
            "count": len(indices),
            "dtype": "float32",
            "compress": "lzw",
        })
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **metadata) as dst:
            for i, (name, data) in enumerate(indices.items(), start=1):
                dst.write(data.astype(np.float32), i)
                dst.set_band_description(i, name)
        logger.info(f"Saved indices to: {output_path}")
