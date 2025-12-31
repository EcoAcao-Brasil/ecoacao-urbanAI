"""
Visualization Components

Generate maps and plots for urban heat analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)


class MapGenerator:
    """
    Generate visualizations for urban heat analysis.
    
    Args:
        output_dir: Directory for saving visualizations
        dpi: Resolution for saved figures
        cmap: Default colormap
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
        
        logger.info(f"MapGenerator initialized: {output_dir}")
    
    def plot_temporal_evolution(
        self,
        data_dir: Path,
        metric: str = "LST",
        save_name: str = "temporal_evolution.png",
    ) -> Path:
        """
        Plot temporal evolution of a metric.
        
        Args:
            data_dir: Directory with processed features
            metric: Metric to plot (NDBI, NDVI, LST, etc.)
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Plotting temporal evolution for {metric}")
        
        # Find all feature files
        files = sorted(data_dir.glob("*_features*.tif"))
        
        if not files:
            raise ValueError(f"No feature files found in {data_dir}")
        
        years = []
        mean_values = []
        
        # Extract metric from each file
        for file_path in files:
            year = self._extract_year(file_path.name)
            years.append(year)
            
            with rasterio.open(file_path) as src:
                descriptions = list(src.descriptions or [])
                
                if metric in descriptions:
                    band_idx = descriptions.index(metric) + 1
                    data = src.read(band_idx)
                    # Calculate mean, ignoring zeros
                    mean_val = np.mean(data[data != 0])
                    mean_values.append(mean_val)
                else:
                    mean_values.append(np.nan)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(years, mean_values, marker='o', linewidth=2, markersize=8)
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel(f"Mean {metric}", fontsize=12)
        ax.set_title(f"Temporal Evolution of {metric}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Save
        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
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
        Plot prediction as map.
        
        Args:
            prediction_path: Path to prediction raster
            title: Plot title
            save_name: Output filename
            band_name: Band to visualize
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Plotting prediction map: {band_name}")
        
        with rasterio.open(prediction_path) as src:
            descriptions = list(src.descriptions or [])
            
            if band_name in descriptions:
                band_idx = descriptions.index(band_name) + 1
            else:
                band_idx = 5  # Default to LST
            
            data = src.read(band_idx)
            bounds = src.bounds
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot raster
        im = ax.imshow(
            data,
            cmap=self.cmap,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(band_name, fontsize=11)
        
        # Labels
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Save
        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    def plot_intervention_map(
        self,
        priorities_path: Path,
        title: str = "Intervention Priorities",
        save_name: str = "intervention_map.png",
    ) -> Path:
        """
        Plot intervention priority zones.
        
        Args:
            priorities_path: Path to priority zones raster
            title: Plot title
            save_name: Output filename
            
        Returns:
            Path to saved figure
        """
        logger.info("Plotting intervention priority map")
        
        with rasterio.open(priorities_path) as src:
            priority_mask = src.read(1)
            labeled = src.read(2)
            bounds = src.bounds
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot priority zones
        cmap = LinearSegmentedColormap.from_list(
            "priority",
            ["white", "yellow", "orange", "red"],
        )
        
        im = ax.imshow(
            labeled,
            cmap=cmap,
            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        )
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Priority Zone ID", fontsize=11)
        
        # Labels
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add text with statistics
        n_zones = int(labeled.max())
        n_pixels = int(np.sum(priority_mask > 0))
        
        textstr = f"Priority Zones: {n_zones}\nAffected Pixels: {n_pixels:,}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=props,
        )
        
        # Save
        output_path = self.output_dir / save_name
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path}")
        return output_path
    
    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")


__all__ = ["MapGenerator"]
