"""
Intervention Priority Analysis

Identify hotspots and priority zones for urban heat intervention.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
from shapely.geometry import shape

logger = logging.getLogger(__name__)


class InterventionAnalyzer:
    """
    Identify priority zones for urban heat intervention.

    Analyzes residuals to find areas where heat is expected to worsen
    significantly, requiring intervention.

    Args:
        residuals_path: Path to combined residuals raster
        current_raster: Path to current year raster (for context)
        output_dir: Directory for outputs
    """

    def __init__(
        self,
        residuals_path: Path,
        current_raster: Path,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.residuals_path = Path(residuals_path)
        self.current_raster = Path(current_raster)
        self.output_dir = Path(output_dir) if output_dir else Path("analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("InterventionAnalyzer initialized")

    def identify_priority_zones(
        self,
        threshold: str = "high",
        min_area_pixels: int = 10,
        save_geojson: bool = True,
        save_raster: bool = True,
    ) -> Dict:
        """
        Identify priority intervention zones.

        Args:
            threshold: Priority threshold ('low', 'medium', 'high')
            min_area_pixels: Minimum area for valid zone
            save_geojson: Save as GeoJSON for GIS
            save_raster: Save as raster

        Returns:
            Dictionary with results and statistics
        """
        logger.info(f"Identifying priority zones (threshold: {threshold})")

        # Load residuals
        with rasterio.open(self.residuals_path) as src:
            residuals = src.read(1)
            metadata = src.meta.copy()
            transform = src.transform
            crs = src.crs

        # Load current LST for context
        current_lst = self._load_current_lst()

        # Define thresholds based on residual intensity
        threshold_values = {
            "low": 0.3,      # 30th percentile of positive residuals
            "medium": 0.5,   # 50th percentile
            "high": 0.7,     # 70th percentile (most severe)
        }
        thresh = threshold_values.get(threshold, 0.5)

        # Identify hotspots (positive residuals above threshold)
        # Positive residuals = areas getting HOTTER
        priority_mask = residuals > thresh

        logger.info(f"Initial priority pixels: {np.sum(priority_mask):,}")

        # Filter by minimum area using connected components
        labeled, n_zones = ndimage.label(priority_mask)
        sizes = ndimage.sum(priority_mask, labeled, range(n_zones + 1))

        # Remove zones smaller than minimum area
        remove_small = sizes < min_area_pixels
        remove_small_mask = remove_small[labeled]
        priority_mask[remove_small_mask] = False

        # Re-label after filtering
        labeled, n_zones = ndimage.label(priority_mask)

        logger.info(f"Found {n_zones} priority zones after filtering")

        # Calculate statistics for each zone
        stats = self._calculate_zone_statistics(
            priority_mask,
            residuals,
            current_lst,
            labeled,
            n_zones,
        )

        # Save outputs
        output_paths = {}

        if save_raster:
            raster_path = self._save_priority_raster(
                priority_mask,
                labeled,
                metadata,
            )
            output_paths["raster"] = raster_path

        if save_geojson:
            geojson_path = self._save_priority_geojson(
                labeled,
                stats,
                transform,
                crs,
            )
            output_paths["geojson"] = geojson_path

        # Save statistics to CSV
        csv_path = self._save_statistics_csv(stats)
        output_paths["statistics_csv"] = csv_path

        results = {
            "n_priority_pixels": int(np.sum(priority_mask)),
            "n_hotspot_zones": n_zones,
            "threshold_used": threshold,
            "threshold_value": thresh,
            "statistics": stats,
            "output_path": output_paths.get("raster"),
            "geojson_path": output_paths.get("geojson"),
            "csv_path": csv_path,
        }

        logger.info(f"Analysis complete:")
        logger.info(f"  Priority pixels: {results['n_priority_pixels']:,}")
        logger.info(f"  Hotspot zones: {results['n_hotspot_zones']}")

        return results

    def _load_current_lst(self) -> Optional[np.ndarray]:
        """Load current LST band for context."""
        try:
            with rasterio.open(self.current_raster) as src:
                descriptions = src.descriptions or []
                if "LST" in descriptions:
                    lst_idx = descriptions.index("LST") + 1
                    return src.read(lst_idx)
        except Exception as e:
            logger.warning(f"Could not load current LST: {str(e)}")
        
        return None

    def _calculate_zone_statistics(
        self,
        priority_mask: np.ndarray,
        residuals: np.ndarray,
        current_lst: Optional[np.ndarray],
        labeled: np.ndarray,
        n_zones: int,
    ) -> List[Dict]:
        """Calculate statistics for each priority zone."""
        stats = []

        for zone_id in range(1, n_zones + 1):
            zone_mask = labeled == zone_id

            zone_stats = {
                "zone_id": zone_id,
                "area_pixels": int(np.sum(zone_mask)),
                "area_m2": int(np.sum(zone_mask) * 900),  # Assuming 30m pixels
                "mean_residual": float(np.mean(residuals[zone_mask])),
                "max_residual": float(np.max(residuals[zone_mask])),
                "std_residual": float(np.std(residuals[zone_mask])),
            }

            # Add current LST if available
            if current_lst is not None:
                zone_stats["current_mean_lst"] = float(np.mean(current_lst[zone_mask]))
                zone_stats["current_max_lst"] = float(np.max(current_lst[zone_mask]))

            # Calculate centroid
            coords = np.argwhere(zone_mask)
            centroid = coords.mean(axis=0)
            zone_stats["centroid_row"] = int(centroid[0])
            zone_stats["centroid_col"] = int(centroid[1])

            # Calculate compactness (perimeter² / area)
            perimeter = np.sum(ndimage.binary_dilation(zone_mask) ^ zone_mask)
            area = np.sum(zone_mask)
            zone_stats["compactness"] = float((perimeter ** 2) / area if area > 0 else 0)

            stats.append(zone_stats)

        # Sort by severity (mean residual * area)
        stats.sort(
            key=lambda x: x["mean_residual"] * x["area_pixels"],
            reverse=True
        )

        # Add priority ranking
        for rank, zone_stat in enumerate(stats, start=1):
            zone_stat["priority_rank"] = rank

        return stats

    def _save_priority_raster(
        self,
        priority_mask: np.ndarray,
        labeled: np.ndarray,
        metadata: Dict,
    ) -> Path:
        """Save priority zones as multi-band raster."""
        output_path = self.output_dir / "intervention_priorities.tif"

        meta = metadata.copy()
        meta.update({
            "count": 2,
            "dtype": "uint16",
            "compress": "lzw",
        })

        with rasterio.open(output_path, "w", **meta) as dst:
            # Band 1: Binary priority mask
            dst.write(priority_mask.astype(np.uint16), 1)
            dst.set_band_description(1, "Priority_Mask")
            
            # Band 2: Zone labels
            dst.write(labeled.astype(np.uint16), 2)
            dst.set_band_description(2, "Zone_Labels")

        logger.info(f"Saved priority raster: {output_path}")
        return output_path

    def _save_priority_geojson(
        self,
        labeled: np.ndarray,
        stats: List[Dict],
        transform,
        crs,
    ) -> Path:
        """Save priority zones as GeoJSON for GIS."""
        output_path = self.output_dir / "intervention_priorities.geojson"

        # Create statistics lookup
        stats_lookup = {s["zone_id"]: s for s in stats}

        # Vectorize zones
        geometries = []
        properties = []

        for geom, value in shapes(labeled.astype(np.int32), transform=transform):
            zone_id = int(value)
            if zone_id > 0:  # Skip background
                geometries.append(shape(geom))
                properties.append(stats_lookup.get(zone_id, {"zone_id": zone_id}))

        if not geometries:
            logger.warning("No geometries to save")
            return output_path

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)

        # Save as GeoJSON
        gdf.to_file(output_path, driver="GeoJSON")

        logger.info(f"Saved priority GeoJSON: {output_path}")
        return output_path

    def _save_statistics_csv(self, stats: List[Dict]) -> Path:
        """Save zone statistics to CSV."""
        output_path = self.output_dir / "priority_zones_statistics.csv"

        import pandas as pd
        df = pd.DataFrame(stats)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved statistics CSV: {output_path}")
        return output_path

    def generate_summary_report(self) -> str:
        """
        Generate text summary report.

        Returns:
            Formatted summary text
        """
        # Load priority zones
        try:
            with rasterio.open(self.output_dir / "intervention_priorities.tif") as src:
                priority_mask = src.read(1)
                labeled = src.read(2)
        except:
            return "No priority zones found. Run identify_priority_zones() first."

        n_pixels = np.sum(priority_mask > 0)
        n_zones = np.max(labeled)
        total_pixels = priority_mask.size

        report = f"""
URBAN HEAT INTERVENTION PRIORITY REPORT, BY ECOAÇÃO BRASIL (BRAZIL ECOACTION):

*SUMMARY*
Total Priority Pixels:    {n_pixels:,} ({n_pixels/total_pixels*100:.2f}%)
Number of Hotspot Zones:  {n_zones}
Average Zone Size:        {n_pixels/n_zones if n_zones > 0 else 0:.1f} pixels

*RECOMMENDATIONS*
1. Focus interventions on highest-ranked zones first
2. Prioritize zones with high current LST + high residual
3. Consider zone compactness for intervention feasibility
4. Review GeoJSON in GIS for spatial planning

*OUTPUT FILES*

- intervention_priorities.tif (raster)
- intervention_priorities.geojson (vector)
- priority_zones_statistics.csv (data)

"""
        return report
