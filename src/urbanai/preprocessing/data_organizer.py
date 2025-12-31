"""
Temporal Data Organization

Orchestrates complete data preprocessing workflow.
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

    Handles:
    1. Spectral indices calculation (NDBI, NDVI, NDWI, NDBSI, LST)
    2. Tocantins Framework integration (IS, SS)
    3. Temporal sequence organization

    Args:
        raw_dir: Directory with raw Landsat GeoTIFFs
        output_dir: Directory for processed outputs
        config: Processing configuration
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
        self.band_processor = MultiTemporalProcessor(
            landsat_version=self.config.get("landsat_version", 8)
        )

        self.tocantins_processor = BatchTocantinsProcessor(
            k_threshold=self.config.get("k_threshold", 1.5),
            spatial_params=self.config.get("spatial_params"),
            rf_params=self.config.get("rf_params"),
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

        # Filter by years if specified
        if years:
            available_files = [
                f for f in available_files
                if self._extract_year(f.name) in years
            ]
            logger.info(f"Filtered to {len(available_files)} files for specified years")

        # Step 1: Calculate spectral indices
        if calculate_indices:
            logger.info("Step 1/2: Calculating spectral indices...")

            indices_dir = self.output_dir / "indices"
            indices_results = self.band_processor.process_time_series(
                input_dir=self.raw_dir,
                output_dir=indices_dir,
                pattern="*_cropped.tif",
            )

            logger.info(f"Processed {len(indices_results)} years")

        # Step 2: Calculate Tocantins scores
        if calculate_tocantins:
            logger.info("Step 2/2: Calculating Tocantins scores...")

            # Use indices as input for Tocantins
            if calculate_indices:
                tocantins_input = indices_dir
            else:
                tocantins_input = self.raw_dir

            tocantins_results = self.tocantins_processor.process_time_series(
                input_dir=tocantins_input,
                output_dir=self.output_dir,
                pattern="*_features.tif",
                save_intermediate=False,
            )

            logger.info(f"Calculated scores for {len(tocantins_results)} years")

        # Final outputs are complete features
        final_files = {}
        for path in sorted(self.output_dir.glob("*_features_complete.tif")):
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
        """
        Process single year.

        Args:
            year: Year to process
            calculate_indices: Calculate spectral indices
            calculate_tocantins: Calculate IS and SS

        Returns:
            Path to output file
        """
        results = self.process_all_years(
            years=[year],
            calculate_indices=calculate_indices,
            calculate_tocantins=calculate_tocantins,
        )

        if year not in results:
            raise ValueError(f"Processing failed for year {year}")

        return results[year]

    def validate_outputs(self) -> bool:
        """
        Validate all processed outputs.

        Returns:
            True if all outputs are valid
        """
        logger.info("Validating processed outputs...")

        output_files = sorted(self.output_dir.glob("*_features_complete.tif"))

        if not output_files:
            logger.error("No output files found")
            return False

        # Check each file
        for file_path in output_files:
            try:
                import rasterio
                with rasterio.open(file_path) as src:
                    # Check band count
                    if src.count != 7:
                        logger.error(f"Invalid band count in {file_path}: {src.count}")
                        return False

                    # Check descriptions
                    expected = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST", "IS", "SS"]
                    descriptions = src.descriptions or []

                    if len(descriptions) != 7:
                        logger.warning(f"Missing band descriptions in {file_path}")

            except Exception as e:
                logger.error(f"Error validating {file_path}: {str(e)}")
                return False

        logger.info(f"All {len(output_files)} files validated successfully.")
        return True

    def get_temporal_statistics(self) -> Dict:
        """
        Get statistics about processed temporal data.

        Returns:
            Dictionary with temporal statistics
        """
        output_files = sorted(self.output_dir.glob("*_features_complete.tif"))
        years = [self._extract_year(f.name) for f in output_files]

        if not years:
            return {}

        stats = {
            "n_years": len(years),
            "start_year": min(years),
            "end_year": max(years),
            "years": sorted(years),
        }

        # Calculate temporal interval
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
    def _default_config() -> Dict:
        """Default processing configuration."""
        return {
            "landsat_version": 8,
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
        }
