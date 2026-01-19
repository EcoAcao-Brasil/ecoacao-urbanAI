"""
Temporal Data Organization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from urbanai.preprocessing.band_processor import MultiTemporalProcessor
from urbanai.preprocessing.tocantins_integration import BatchTocantinsProcessor

logger = logging.getLogger(__name__)


class TemporalDataProcessor:
    """
    Complete temporal data processing orchestrator.

    Workflow:
    1. Calculate spectral indices from RAW Landsat → features.tif
    2. Calculate IS/SS from RAW Landsat → merge into features_complete.tif
    """

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        config: Optional[Dict] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or self._default_config()

        # Initialize processors
        landsat_version = self.config.get("landsat_version", 8)
        band_mapping = self._get_band_mapping(landsat_version)
        self.band_processor = MultiTemporalProcessor(
            band_mapping=band_mapping
        )

        self.tocantins_processor = BatchTocantinsProcessor(
            k_threshold=self.config.get("tocantins", {}).get("k_threshold", 1.5),
            spatial_params=self.config.get("tocantins", {}).get("spatial_params"),
            rf_params=self.config.get("tocantins", {}).get("rf_params"),
        )

        logger.info("TemporalDataProcessor initialized")
        logger.info(f"Raw directory: {self.raw_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def process_all_years(
        self,
        years: Optional[List[int]] = None,
        calculate_indices: bool = True,
        calculate_tocantins: bool = True,
    ) -> Dict[int, Path]:
        """
        Process all years in sequence.

        Args:
            years: List of years to process (if None, processes all available)
            calculate_indices: Calculate spectral indices
            calculate_tocantins: Calculate IS and SS

        Returns:
            Dictionary mapping year to output file path
        """
        logger.info("Starting temporal data processing...")

        # Find available files
        available_files = sorted(self.raw_dir.glob("*_cropped.tif"))
        if not available_files:
            raise ValueError(f"No cropped GeoTIFF files found in {self.raw_dir}")

        logger.info(f"Found {len(available_files)} raw files")

        # Step 1: Calculate spectral indices from RAW files
        indices_dir = self.output_dir / "indices"
        
        if calculate_indices:
            logger.info("Step 1/2: Calculating spectral indices from RAW files...")

            indices_results = self.band_processor.process_time_series(
                input_dir=self.raw_dir,
                output_dir=indices_dir,
                pattern="*_cropped.tif",
            )

            logger.info(f"Processed {len(indices_results)} years")
        else:
            # If not calculating, assume they exist
            if not indices_dir.exists():
                raise ValueError("Indices directory doesn't exist and calculate_indices=False")

        # Step 2: Calculate Tocantins scores from RAW files (not features!)
        if calculate_tocantins:
            logger.info("Step 2/2: Calculating Tocantins scores from RAW files...")

            tocantins_results = self.tocantins_processor.process_time_series(
                raw_dir=self.raw_dir,           # ← RAW files for Tocantins
                features_dir=indices_dir,        # ← Features to merge with
                output_dir=self.output_dir,
                save_intermediate=False,
            )

            logger.info(f"Calculated scores for {len(tocantins_results)} years")

        # Determine final output files based on whether Tocantins was calculated
        final_files = {}
        if calculate_tocantins:
            # Final outputs are complete features (7 bands: NDBI, NDVI, NDWI, NDBSI, LST, IS, SS)
            for path in sorted(self.output_dir.glob("*_features_complete.tif")):
                year = self._extract_year(path.name)
                final_files[year] = path
        else:
            # Final outputs are basic features (5 bands: NDBI, NDVI, NDWI, NDBSI, LST)
            for path in sorted(indices_dir.glob("*_features.tif")):
                year = self._extract_year(path.name)
                final_files[year] = path

        logger.info(f"Processing complete: {len(final_files)} years ready.")

        return final_files

    def process_single_year(
        self,
        year: int,
        calculate_indices: bool = True,
        calculate_tocantins: bool = True,
    ) -> Path:
        """Process single year."""
        results = self.process_all_years(
            years=[year],
            calculate_indices=calculate_indices,
            calculate_tocantins=calculate_tocantins,
        )

        if year not in results:
            raise ValueError(f"Processing failed for year {year}")

        return results[year]

    def validate_outputs(self) -> bool:
        """Validate all processed outputs."""
        logger.info("Validating processed outputs...")

        # Determine expected band count based on config
        tocantins_enabled = self.config.get("tocantins", {}).get("enabled", False)
        expected_bands = 7 if tocantins_enabled else 5
        
        # Determine which files to validate
        if tocantins_enabled:
            output_files = sorted(self.output_dir.glob("*_features_complete.tif"))
            expected_names = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST", "IS", "SS"]
        else:
            indices_dir = self.output_dir / "indices"
            output_files = sorted(indices_dir.glob("*_features.tif"))
            expected_names = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]

        if not output_files:
            logger.error("No output files found")
            return False

        for file_path in output_files:
            try:
                import rasterio
                with rasterio.open(file_path) as src:
                    if src.count != expected_bands:
                        logger.error(
                            f"Invalid band count in {file_path}: {src.count}, "
                            f"expected {expected_bands}"
                        )
                        return False

                    descriptions = src.descriptions or []

                    if len(descriptions) != expected_bands:
                        logger.warning(f"Missing band descriptions in {file_path}")

            except Exception as e:
                logger.error(f"Error validating {file_path}: {str(e)}")
                return False

        logger.info(f"All {len(output_files)} files validated successfully.")
        return True

    def get_temporal_statistics(self) -> Dict:
        """Get statistics about processed temporal data."""
        # Check for both complete and basic features
        tocantins_enabled = self.config.get("tocantins", {}).get("enabled", False)
        
        if tocantins_enabled:
            output_files = sorted(self.output_dir.glob("*_features_complete.tif"))
        else:
            indices_dir = self.output_dir / "indices"
            output_files = sorted(indices_dir.glob("*_features.tif"))
            
        years = [self._extract_year(f.name) for f in output_files]

        if not years:
            return {}

        stats = {
            "n_years": len(years),
            "start_year": min(years),
            "end_year": max(years),
            "years": sorted(years),
        }

        if len(years) >= 2:
            intervals = [years[i+1] - years[i] for i in range(len(years)-1)]
            stats["interval"] = min(intervals)
            stats["interval_max"] = max(intervals)

        return stats

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")

    @staticmethod
    def _get_band_mapping(landsat_version: int) -> Dict[str, str]:
        """
        Get band mapping for specified Landsat version.
        
        Args:
            landsat_version: Landsat satellite version (5 or 8)
            
        Returns:
            Dictionary mapping common band names to Landsat band descriptions
            
        Note:
            Landsat 5 TM has no SR_B6 in the surface reflectance product.
            Band 6 is only available as thermal (ST_B6), so swir2 maps to SR_B7.
        """
        if landsat_version == 8 or landsat_version == 9:
            return {
                "blue": "SR_B2",
                "green": "SR_B3",
                "red": "SR_B4",
                "nir": "SR_B5",
                "swir1": "SR_B6",
                "swir2": "SR_B7",
                "thermal": "ST_B10"
            }
        elif landsat_version == 5 or landsat_version == 7:
            return {
                "blue": "SR_B1",
                "green": "SR_B2",
                "red": "SR_B3",
                "nir": "SR_B4",
                "swir1": "SR_B5",
                "swir2": "SR_B7",
                "thermal": "ST_B6"
            }
        else:
            raise ValueError(
                f"Unsupported Landsat version: {landsat_version}. "
                f"Supported versions: 5, 8"
            )

    @staticmethod
    def _default_config() -> Dict:
        """Default processing configuration."""
        return {
            "landsat_version": 8,
            "tocantins": {
                "k_threshold": 1.5,
                "spatial_params": {
                    "min_anomaly_size": 1,
                    "agglutination_distance": 4,
                    "morphology_kernel_size": 3,
                    "connectivity": 2,
                },
                "rf_params": {
                    "n_estimators": 200,
                    "max_depth": 25,
                    "min_samples_split": 8,
                    "min_samples_leaf": 4,
                    "random_state": 42,
                },
            },
        }
