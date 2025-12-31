"""File Writers"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

logger = logging.getLogger(__name__)


class RasterWriter:
    """Write raster files."""

    @staticmethod
    def write(
        path: Path,
        data: np.ndarray,
        metadata: Dict,
        descriptions: Optional[List[str]] = None,
        compress: str = "lzw",
    ) -> None:
        """
        Write raster file.

        Args:
            path: Output path
            data: Raster data (channels, height, width)
            metadata: Raster metadata (from RasterReader or manual)
            descriptions: Band descriptions
            compress: Compression method
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data is 3D
        if data.ndim == 2:
            data = data[np.newaxis, :, :]

        # Prepare metadata
        meta = {
            "driver": "GTiff",
            "count": data.shape[0],
            "height": data.shape[1],
            "width": data.shape[2],
            "dtype": str(data.dtype),
            "crs": metadata.get("crs"),
            "transform": metadata.get("transform"),
            "compress": compress,
        }

        # Add nodata if specified
        if "nodata" in metadata:
            meta["nodata"] = metadata["nodata"]

        # Write raster
        with rasterio.open(path, "w", **meta) as dst:
            dst.write(data)

            # Set band descriptions
            if descriptions:
                for i, desc in enumerate(descriptions, start=1):
                    dst.set_band_description(i, desc)

        logger.info(f"Wrote raster: {path} ({data.shape})")

    @staticmethod
    def write_array(
        path: Path,
        data: np.ndarray,
        transform,
        crs: str,
        descriptions: Optional[List[str]] = None,
    ) -> None:
        """Write array with transform and CRS."""
        metadata = {
            "transform": transform,
            "crs": crs,
        }
        RasterWriter.write(path, data, metadata, descriptions)


class CSVWriter:
    """Write CSV files."""

    @staticmethod
    def write(
        path: Path,
        data: pd.DataFrame,
        index: bool = False,
    ) -> None:
        """Write DataFrame to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=index)
        logger.info(f"Wrote CSV: {path} ({len(data)} rows)")


class ResultsWriter:
    """Write analysis results."""

    @staticmethod
    def write_predictions(
        output_dir: Path,
        predictions: np.ndarray,
        metadata: Dict,
        year: int,
        descriptions: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Write prediction results.

        Args:
            output_dir: Output directory
            predictions: Predicted data
            metadata: Raster metadata
            year: Prediction year
            descriptions: Band descriptions

        Returns:
            Dictionary of output paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Main prediction
        pred_path = output_dir / f"{year}_predicted.tif"
        RasterWriter.write(pred_path, predictions, metadata, descriptions)
        outputs["prediction"] = pred_path

        return outputs

    @staticmethod
    def write_analysis(
        output_dir: Path,
        residuals: Dict[str, np.ndarray],
        priorities: Dict,
        metadata: Dict,
    ) -> Dict[str, Path]:
        """
        Write analysis results.

        Args:
            output_dir: Output directory
            residuals: Dictionary of residual arrays
            priorities: Priority zone data
            metadata: Raster metadata

        Returns:
            Dictionary of output paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        outputs = {}

        # Residuals
        for name, data in residuals.items():
            path = output_dir / f"{name}_residual.tif"
            RasterWriter.write(path, data, metadata)
            outputs[f"residual_{name}"] = path

        # Statistics CSV
        if "statistics" in priorities:
            stats_path = output_dir / "statistics.csv"
            df = pd.DataFrame(priorities["statistics"])
            CSVWriter.write(stats_path, df)
            outputs["statistics"] = stats_path

        return outputs
