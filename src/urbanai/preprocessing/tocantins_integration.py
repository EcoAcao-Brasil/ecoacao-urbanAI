"""
Tocantins Framework Integration

Calculates Impact Score (IS) and Severity Score (SS) using the Tocantins Framework.
"""

import logging
import re
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

    Calculates Impact Score (IS) and Severity Score (SS) from processed
    Landsat imagery with spectral indices.

    Args:
        k_threshold: Residual threshold multiplier for anomaly detection
        spatial_params: Spatial processing parameters
        rf_params: Random Forest parameters
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
        input_path: Path,
        output_dir: Optional[Path] = None,
        save_intermediate: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Calculates IS and SS for a raster with spectral indices.

        Args:
            input_path: Path to GeoTIFF with bands [NDBI, NDVI, NDWI, NDBSI, LST]
            output_dir: Directory to save outputs
            save_intermediate: Save intermediate Tocantins outputs

        Returns:
            Tuple of (impact_scores, severity_scores, metadata)
        """
        logger.info(f"Calculating Tocantins scores for: {input_path.name}")

        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Read input raster
        with rasterio.open(input_path) as src:
            metadata = src.meta.copy()
            bands = self._read_bands(src)

        # Validate required bands
        required = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]
        missing = [b for b in required if b not in bands]
        if missing:
            raise ValueError(f"Missing required bands: {missing}")

        # Initialize Tocantins calculator
        calculator = TocantinsFrameworkCalculator(
            k_threshold=self.k_threshold,
            spatial_params=self.spatial_params,
            rf_params=self.rf_params,
        )

        # Create temporary GeoTIFF with required format for Tocantins
        temp_path = self._create_tocantins_input(bands, metadata, input_path)

        try:
            # Run Tocantins Framework
            success = calculator.run_complete_analysis(
                tif_path=str(temp_path),
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

        finally:
            # Cleanup temporary file
            if temp_path.exists():
                temp_path.unlink()

    def _read_bands(self, src: rasterio.DatasetReader) -> Dict[str, np.ndarray]:
        """Reads bands from raster."""
        bands = {}
        descriptions = src.descriptions or []

        # Map band descriptions to data
        for i, desc in enumerate(descriptions, start=1):
            if desc in ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]:
                bands[desc] = src.read(i).astype(np.float32)

        return bands

    def _create_tocantins_input(
        self,
        bands: Dict[str, np.ndarray],
        metadata: Dict,
        original_path: Path,
    ) -> Path:
        """
        Creates temporary GeoTIFF in Tocantins-compatible format.

        Tocantins expects: SR_B1-B7, ST_B10, and calculated indices.
        """
        temp_path = original_path.parent / f"temp_tocantins_{original_path.name}"

        # Update metadata
        meta = metadata.copy()
        meta.update({
            "count": len(bands),
            "dtype": "float32",
        })

        # Write bands in expected order
        band_order = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]

        with rasterio.open(temp_path, "w", **meta) as dst:
            for i, band_name in enumerate(band_order, start=1):
                if band_name in bands:
                    dst.write(bands[band_name], i)
                    dst.set_band_description(i, band_name)

        return temp_path

    def _extract_impact_scores(
        self,
        calculator: TocantinsFrameworkCalculator,
    ) -> np.ndarray:
        """Extracts Impact Scores from the Tocantins calculator."""
        try:
            # Retrieve feature set containing Impact Scores
            feature_set = calculator.get_feature_set()

            if feature_set is None or feature_set.empty:
                logger.warning("Tocantins: No features calculated. Returning zero array.")
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)

            # Get spatial dimensions
            classification_map = calculator.get_classification_map()
            height, width = classification_map.shape

            # Initialize Impact Score raster
            is_raster = np.zeros((height, width), dtype=np.float32)

            # Map Impact Score values to spatial coordinates
            for _, row in feature_set.iterrows():
                if "impact_score" in row and "pixels" in row:
                    pixels = row["pixels"]
                    if isinstance(pixels, list) and pixels:
                        for pixel in pixels:
                            y, x = pixel
                            if 0 <= y < height and 0 <= x < width:
                                is_raster[y, x] = row["impact_score"]

            # Validate population of raster values
            if np.max(is_raster) == 0:
                logger.warning("Tocantins: Impact Score calculation produced all zeros.")
            else:
                min_val = np.min(is_raster[is_raster > 0])
                max_val = np.max(is_raster)
                logger.info(f"Impact Scores calculated successfully. Range: [{min_val:.3f}, {max_val:.3f}]")

            return is_raster

        except Exception as e:
            logger.error(f"Tocantins Impact Score extraction failed: {e}")
            logger.warning("Returning zero array for Impact Scores due to extraction failure.")
            
            # Attempt fallback to zero array; raise critical error if dimension retrieval fails
            try:
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)
            except Exception as critical_error:
                raise TocantinsCalculationError(f"Critical preprocessing failure: {critical_error}") from e

    def _extract_severity_scores(
        self,
        calculator: TocantinsFrameworkCalculator,
    ) -> np.ndarray:
        """Extracts Severity Scores from the Tocantins calculator."""
        try:
            # Retrieve severity scores dataframe
            severity_df = calculator.get_severity_scores()

            if severity_df is None or severity_df.empty:
                logger.warning("Tocantins: No severity scores calculated. Returning zero array.")
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)

            # Get spatial dimensions
            classification_map = calculator.get_classification_map()
            height, width = classification_map.shape

            # Initialize Severity Score raster
            ss_raster = np.zeros((height, width), dtype=np.float32)

            # Map Severity Score values to spatial coordinates
            for _, row in severity_df.iterrows():
                if "severity_score" in row and "core_pixels" in row:
                    pixels = row["core_pixels"]
                    if isinstance(pixels, list) and pixels:
                        for pixel in pixels:
                            y, x = pixel
                            if 0 <= y < height and 0 <= x < width:
                                ss_raster[y, x] = row["severity_score"]

            # Validate population of raster values
            if np.max(ss_raster) == 0:
                logger.warning("Tocantins: Severity Score calculation produced all zeros.")
            else:
                min_val = np.min(ss_raster[ss_raster > 0])
                max_val = np.max(ss_raster)
                logger.info(f"Severity Scores calculated successfully. Range: [{min_val:.3f}, {max_val:.3f}]")

            return ss_raster

        except Exception as e:
            logger.error(f"Tocantins Severity Score extraction failed: {e}")
            logger.warning("Returning zero array for Severity Scores due to extraction failure.")
            
            # Attempt fallback to zero array; raise critical error if dimension retrieval fails
            try:
                classification_map = calculator.get_classification_map()
                return np.zeros_like(classification_map, dtype=np.float32)
            except Exception as critical_error:
                raise TocantinsCalculationError(f"Cannot extract severity scores: {critical_error}") from e

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
    Process multiple rasters with Tocantins Framework.

    Handles batch processing of temporal sequences.
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
        input_dir: Path,
        output_dir: Path,
        pattern: str = "*_features.tif",
        save_intermediate: bool = False,
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        Process all rasters in directory with Tocantins.

        Args:
            input_dir: Directory with processed feature rasters
            output_dir: Directory for Tocantins outputs
            pattern: Filename pattern
            save_intermediate: Save intermediate results

        Returns:
            Dictionary mapping year to (IS, SS) arrays
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = sorted(input_dir.glob(pattern))
        if not input_files:
            raise ValueError(f"No files matching {pattern} in {input_dir}")

        logger.info(f"Processing {len(input_files)} rasters with Tocantins Framework.")

        results = {}
        for input_path in input_files:
            year = self._extract_year(input_path.name)

            logger.info(f"Processing year: {year}")

            # Calculate scores
            year_output_dir = output_dir / str(year) if save_intermediate else None

            is_raster, ss_raster, stats = self.integration.calculate_scores(
                input_path=input_path,
                output_dir=year_output_dir,
                save_intermediate=save_intermediate,
            )

            results[year] = (is_raster, ss_raster)

            # Save combined raster
            self._save_combined(
                input_path, is_raster, ss_raster, output_dir, year
            )

        logger.info(f"Completed Tocantins processing for {len(results)} years.")
        return results

    def _save_combined(
        self,
        input_path: Path,
        is_raster: np.ndarray,
        ss_raster: np.ndarray,
        output_dir: Path,
        year: int,
    ) -> None:
        """Saves combined raster with all features + IS + SS."""
        with rasterio.open(input_path) as src:
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

        logger.info(f"Saved complete features to: {output_path}")

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extracts year from filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")
