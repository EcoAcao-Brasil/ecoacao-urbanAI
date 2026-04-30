"""
Visualization Components

Generate PNG maps and temporal evolution plots from raster data.
"""

import logging
import re
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import rasterio

logger = logging.getLogger(__name__)


class MapGenerator:
    """
    Generate maps and temporal plots from processed and predicted rasters.

    Args:
        output_dir: Directory for saving output PNGs.
        dpi: Resolution for saved figures.
        cmap: Default colormap.
    """

    def __init__(
        self,
        output_dir: Path,
        dpi: int = 300,
        cmap: str = "RdYlBu_r",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.cmap = cmap

    def plot_temporal_evolution(
        self,
        data_dir: Path,
        metric: str = "LST",
        save_name: str = "temporal_evolution.png",
    ) -> Path:
        """
        Plot the mean value of a metric across all available years.

        Args:
            data_dir: Directory containing processed feature rasters.
            metric: Band name to plot (e.g. 'LST', 'NDVI').
            save_name: Output filename.

        Returns:
            Path to saved figure.
        """
        files = sorted(data_dir.glob("*_features*.tif"))
        if not files:
            raise ValueError(f"No feature files found in {data_dir}")

        years = []
        mean_values = []

        for file_path in files:
            year = self._extract_year(file_path.name)
            years.append(year)

            with rasterio.open(file_path) as src:
                descriptions = list(src.descriptions or [])
                if metric in descriptions:
                    band_idx = descriptions.index(metric) + 1
                    data = src.read(band_idx)
                    mean_val = np.mean(data[data != 0])
                    mean_values.append(mean_val)
                else:
                    mean_values.append(np.nan)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(years, mean_values, marker="o", linewidth=2, markersize=8)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(f"Mean {metric}", fontsize=12)
        ax.set_title(f"Temporal Evolution of {metric}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    def plot_prediction_map(
        self,
        prediction_path: Path,
        title: str = "Predicted Urban Heat",
        save_name: str = "prediction_map.png",
        band_name: str = "LST",
    ) -> Path:
        """
        Plot a single band from a predicted raster as a map.

        Args:
            prediction_path: Path to the predicted GeoTIFF.
            title: Plot title.
            save_name: Output filename.
            band_name: Band to visualize.

        Returns:
            Path to saved figure.
        """
        with rasterio.open(prediction_path) as src:
            descriptions = list(src.descriptions or [])
            band_idx = descriptions.index(band_name) + 1 if band_name in descriptions else 1
            data = src.read(band_idx)
            bounds = src.bounds

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(
            data,
            cmap=self.cmap,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        )
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(band_name, fontsize=11)
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold")

        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved: {output_path}")
        return output_path

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract a 4-digit year from a filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")


__all__ = ["MapGenerator"]
