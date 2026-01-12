"""
Tocantins Framework Integration with Resume Support

1. Tocantins processes RAW Landsat files, not feature files
2. Batch processor tracks completed years
3. Results are merged with existing feature files
"""

import logging
import re
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from tocantins_framework import TocantinsFrameworkCalculator

logger = logging.getLogger(__name__)


class TocantinsCalculationError(Exception):
    """Exception for Tocantins calculation failures."""
    pass


class TocantinsIntegration:
    """
    Integration with Tocantins Framework for thermal anomaly detection.
    
    IMPORTANT: Tocantins requires RAW Landsat GeoTIFFs with all bands.
    Do NOT pass processed feature files.
    """

    def __init__(
        self,
        k_threshold: float = 1.5,
        spatial_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
    ) -> None:
        self.k_threshold = k_threshold
        self.spatial_params = spatial_params or self._default_spatial_params()
        self.rf_params = rf_params or self._default_rf_params()

        logger.info("TocantinsIntegration initialized.")
        logger.info(f"Configuration - k_threshold: {k_threshold}")

    def calculate_scores(
        self,
        raw_landsat_path: Path,
        output_dir: Optional[Path] = None,
        save_intermediate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calculates IS and SS from RAW Landsat GeoTIFF.

        Args:
            raw_landsat_path: Path to RAW Landsat GeoTIFF (with SR_B* and ST_B* bands)
            output_dir: Directory to save outputs
            save_intermediate: Save intermediate Tocantins outputs

        Returns:
            Tuple of (impact_scores, severity_scores, metadata)
        """
        logger.info(f"Calculating Tocantins scores for: {raw_landsat_path.name}")

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Tocantins calculator
        calculator = TocantinsFrameworkCalculator(
            k_threshold=self.k_threshold,
            spatial_params=self.spatial_params,
            rf_params=self.rf_params,
        )

        try:
            # Run Tocantins Framework on RAW file
            success = calculator.run_complete_analysis(
                tif_path=str(raw_landsat_path),
                output_dir=str(output_dir) if output_dir else None,
                save_results=save_intermediate,
            )

            if not success:
                raise TocantinsCalculationError("Tocantins Framework calculation failed.")

            # Extract scores
            impact_scores = self._extract_impact_scores(calculator)
            severity_scores = self._extract_severity_scores(calculator)

            # Get statistics
            stats = self._calculate_statistics(impact_scores, severity_scores)

            logger.info(f"Calculated scores - IS Range: [{stats['is_min']:.3f}, {stats['is_max']:.3f}]")
            logger.info(f"                    SS Range: [{stats['ss_min']:.3f}, {stats['ss_max']:.3f}]")

            return impact_scores, severity_scores, stats

        except Exception as e:
            logger.error(f"Tocantins calculation failed: {e}")
            raise

    def _extract_impact_scores(
        self,
        calculator: TocantinsFrameworkCalculator,
    ) -> np.ndarray:
        """Extracts Impact Scores from the Tocantins calculator."""
        try:
            feature_set = calculator.get_feature_set()

            if feature_set is None or feature_set.empty:
                logger.warning("Tocantins: No features calculated. Returning zero array.")
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)

            classification_map = calculator.get_classification_map()
            height, width = classification_map.shape
            is_raster = np.zeros((height, width), dtype=np.float32)

            # Map IS values to pixel locations
            for _, row in feature_set.iterrows():
                if "IS" in row:
                    cy = int(row.get("Centroid_Row", 0))
                    cx = int(row.get("Centroid_Col", 0))
                    
                    if 0 <= cy < height and 0 <= cx < width:
                        is_raster[cy, cx] = row["IS"]

            if np.max(is_raster) == 0:
                logger.warning("Tocantins: Impact Score calculation produced all zeros.")

            return is_raster

        except Exception as e:
            logger.error(f"Impact Score extraction failed: {e}")
            classification_map = calculator.get_classification_map()
            return np.zeros_like(classification_map, dtype=np.float32)

    def _extract_severity_scores(
        self,
        calculator: TocantinsFrameworkCalculator,
    ) -> np.ndarray:
        """Extracts Severity Scores from the Tocantins calculator."""
        try:
            severity_df = calculator.get_severity_scores()

            if severity_df is None or severity_df.empty:
                logger.warning("Tocantins: No severity scores calculated. Returning zero array.")
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)

            classification_map = calculator.get_classification_map()
            height, width = classification_map.shape
            ss_raster = np.zeros((height, width), dtype=np.float32)

            # Map SS values
            for _, row in severity_df.iterrows():
                if "SS" in row:
                    cy = int(row.get("Centroid_Row", 0))
                    cx = int(row.get("Centroid_Col", 0))
                    
                    if 0 <= cy < height and 0 <= cx < width:
                        ss_raster[cy, cx] = row["SS"]

            if np.max(ss_raster) == 0:
                logger.warning("Tocantins: Severity Score calculation produced all zeros.")

            return ss_raster

        except Exception as e:
            logger.error(f"Severity Score extraction failed: {e}")
            classification_map = calculator.get_classification_map()
            return np.zeros_like(classification_map, dtype=np.float32)

    def _calculate_statistics(
        self,
        is_raster: np.ndarray,
        ss_raster: np.ndarray,
    ) -> Dict:
        """Calculates statistics for IS and SS."""
        return {
            "is_min": float(np.min(is_raster)),
            "is_max": float(np.max(is_raster)),
            "is_mean": float(np.mean(is_raster)),
            "is_std": float(np.std(is_raster)),
            "ss_min": float(np.min(ss_raster)),
            "ss_max": float(np.max(ss_raster)),
            "ss_mean": float(np.mean(ss_raster)),
            "ss_std": float(np.std(ss_raster)),
            "n_anomaly_pixels": int(np.sum(is_raster != 0)),
            "n_core_pixels": int(np.sum(ss_raster != 0)),
        }

    @staticmethod
    def _default_spatial_params() -> Dict:
        """Default spatial processing parameters."""
        return {
            "min_anomaly_size": 1,
            "agglutination_distance": 4,
            "morphology_kernel_size": 3,
            "connectivity": 2,
        }

    @staticmethod
    def _default_rf_params() -> Dict:
        """Default Random Forest parameters."""
        return {
            "n_estimators": 200,
            "max_depth": 25,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "random_state": 42,
            "n_jobs": -1,
        }


class BatchTocantinsProcessor:
    """
    Process multiple rasters with Tocantins Framework and resume capability.
    
    CRITICAL: This processes RAW Landsat files, then merges
    IS/SS scores with existing feature files.
    """

    def __init__(
        self,
        k_threshold: float = 1.5,
        spatial_params: Optional[Dict] = None,
        rf_params: Optional[Dict] = None,
    ) -> None:
        self.integration = TocantinsIntegration(
            k_threshold=k_threshold,
            spatial_params=spatial_params,
            rf_params=rf_params,
        )

    def process_time_series(
        self,
        raw_dir: Path,
        features_dir: Path,
        output_dir: Path,
        save_intermediate: bool = False,
        resume: bool = True,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Process RAW Landsat files with Tocantins, then merge with features.

        Args:
            raw_dir: Directory with RAW Landsat GeoTIFFs (*_cropped.tif)
            features_dir: Directory with processed feature rasters (*_features.tif)
            output_dir: Directory for final outputs
            save_intermediate: Save intermediate results
            resume: Skip already-completed years

        Returns:
            Dictionary mapping year to (IS, SS) arrays
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find RAW Landsat files
        raw_files = sorted(raw_dir.glob("*_cropped.tif"))
        if not raw_files:
            raise ValueError(f"No RAW Landsat files (*_cropped.tif) found in {raw_dir}")

        # Progress tracker
        progress_file = output_dir / ".tocantins_progress.json"
        completed_years = self._load_progress(progress_file) if resume else set()

        logger.info(f"Processing {len(raw_files)} RAW Landsat files with Tocantins Framework.")
        if completed_years:
            logger.info(f"Resuming: {len(completed_years)} years already processed")

        results = {}
        for raw_path in raw_files:
            year = self._extract_year(raw_path.name)

            # Check if complete file already exists
            complete_path = output_dir / f"{year}_features_complete.tif"
            if resume and complete_path.exists() and year in completed_years:
                logger.info(f"✓ Skipping {year} (already has complete features)")
                results[year] = (None, None)
                continue

            logger.info(f"Processing Tocantins for {year}...")

            # Calculate IS/SS from RAW file
            year_output_dir = output_dir / str(year) if save_intermediate else None

            try:
                is_raster, ss_raster, stats = self.integration.calculate_scores(
                    raw_landsat_path=raw_path,
                    output_dir=year_output_dir,
                    save_intermediate=save_intermediate,
                )

                results[year] = (is_raster, ss_raster)

                # Merge with existing features
                feature_path = features_dir / f"{year}_features.tif"
                if not feature_path.exists():
                    logger.warning(f"Feature file not found: {feature_path}, skipping merge")
                    continue

                self._save_combined(
                    feature_path, is_raster, ss_raster, output_dir, year
                )

                # Mark as completed
                completed_years.add(year)
                self._save_progress(progress_file, completed_years)

            except Exception as e:
                logger.error(f"Failed to process Tocantins for {year}: {e}")
                continue

        logger.info(f"Completed Tocantins processing for {len(results)} years.")
        return results

    def _save_combined(
        self,
        feature_path: Path,
        is_raster: np.ndarray,
        ss_raster: np.ndarray,
        output_dir: Path,
        year: int,
    ) -> None:
        """Merges IS/SS with existing feature file."""
        with rasterio.open(feature_path) as src:
            meta = src.meta.copy()
            existing_bands = [src.read(i) for i in range(1, src.count + 1)]
            descriptions = list(src.descriptions or [])

        # Add IS and SS
        all_bands = existing_bands + [is_raster, ss_raster]
        descriptions += ["IS", "SS"]

        # Update metadata
        meta.update({"count": len(all_bands)})

        # Save
        output_path = output_dir / f"{year}_features_complete.tif"
        with rasterio.open(output_path, "w", **meta) as dst:
            for i, (band, desc) in enumerate(zip(all_bands, descriptions), start=1):
                dst.write(band.astype(np.float32), i)
                dst.set_band_description(i, desc)

        logger.info(f"✓ Saved complete features: {output_path.name}")

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
        """Extracts year from filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")
