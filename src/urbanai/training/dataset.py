"""
PyTorch Dataset for Spatiotemporal Urban Heat Data

Handles loading and preprocessing of temporal raster sequences.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class UrbanHeatDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal urban heat data.

    Loads temporal sequences of rasters with 7 channels:
    [NDBI, NDVI, NDWI, NDBSI, LST, IS, SS]

    Args:
        data_dir: Directory with processed feature rasters
        sequence_length: Length of input sequences
        prediction_horizon: Number of future steps to predict
        years: List of years to include (if None, uses all available)
        normalize: Whether to normalize features
        augment: Whether to apply data augmentation
        cache_in_memory: Cache all data in memory (faster but more RAM)
    """

    def __init__(
        self,
        data_dir: Path,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        years: Optional[List[int]] = None,
        normalize: bool = True,
        augment: bool = False,
        cache_in_memory: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.augment = augment
        self.cache_in_memory = cache_in_memory

        # Find available files
        self.file_paths = self._find_files(years)
        if len(self.file_paths) < sequence_length + prediction_horizon:
            raise ValueError(
                f"Need at least {sequence_length + prediction_horizon} years, "
                f"found {len(self.file_paths)}"
            )

        # Get spatial dimensions from first file
        with rasterio.open(self.file_paths[0]) as src:
            self.height = src.height
            self.width = src.width
            self.n_channels = src.count

        if self.n_channels != 7:
            raise ValueError(f"Expected 7 channels, found {self.n_channels}")

        # Calculate normalization statistics
        self.stats = self._calculate_stats() if normalize else None

        # Cache data if requested
        self.cache: Dict[int, np.ndarray] = {}
        if cache_in_memory:
            logger.info("Caching all data in memory...")
            for idx, path in enumerate(self.file_paths):
                self.cache[idx] = self._load_raster(path)
            logger.info(f"Cached {len(self.cache)} rasters")

        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")

    def __len__(self) -> int:
        """Number of available sequences."""
        return len(self.file_paths) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: (seq_len, channels, height, width)
            - target_sequence: (pred_horizon, channels, height, width)
        """
        # Get sequence of rasters
        input_indices = list(range(idx, idx + self.sequence_length))
        target_indices = list(
            range(
                idx + self.sequence_length,
                idx + self.sequence_length + self.prediction_horizon,
            )
        )

        # Load data
        if self.cache_in_memory:
            input_data = [self.cache[i] for i in input_indices]
            target_data = [self.cache[i] for i in target_indices]
        else:
            input_data = [self._load_raster(self.file_paths[i]) for i in input_indices]
            target_data = [self._load_raster(self.file_paths[i]) for i in target_indices]

        # Stack into sequences
        input_seq = np.stack(input_data, axis=0)  # (seq_len, channels, h, w)
        target_seq = np.stack(target_data, axis=0)  # (pred_horizon, channels, h, w)

        # Normalize
        if self.normalize:
            input_seq = self._normalize(input_seq)
            target_seq = self._normalize(target_seq)

        # Augment
        if self.augment:
            input_seq, target_seq = self._augment_sequence(input_seq, target_seq)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_seq).float()
        target_tensor = torch.from_numpy(target_seq).float()

        return input_tensor, target_tensor

    def _find_files(self, years: Optional[List[int]]) -> List[Path]:
        """Find and sort feature raster files."""
        pattern = "*_features_complete.tif"
        files = sorted(self.data_dir.glob(pattern))

        if not files:
            # Try alternative pattern
            pattern = "*_features.tif"
            files = sorted(self.data_dir.glob(pattern))

        if not files:
            raise ValueError(f"No feature files found in {self.data_dir}")

        # Filter by years if specified
        if years is not None:
            files = [f for f in files if self._extract_year(f.name) in years]

        # Sort by year
        files = sorted(files, key=lambda f: self._extract_year(f.name))

        logger.info(f"Found {len(files)} raster files")
        return files

    def _load_raster(self, path: Path) -> np.ndarray:
        """
        Load raster file.

        Returns:
            Array of shape (channels, height, width)
        """
        with rasterio.open(path) as src:
            data = src.read()  # (channels, height, width)

        # Handle NaN and infinite values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data.astype(np.float32)

    def _calculate_stats(self) -> Dict[str, np.ndarray]:
        """Calculate mean and std for normalization."""
        logger.info("Calculating normalization statistics...")

        all_data = []
        for path in self.file_paths:
            data = self._load_raster(path)
            # Reshape to (channels, -1) for easier computation
            data_flat = data.reshape(data.shape[0], -1)
            all_data.append(data_flat)

        # Concatenate all data
        all_data = np.concatenate(all_data, axis=1)  # (channels, total_pixels)

        # Calculate per-channel statistics
        mean = np.mean(all_data, axis=1, keepdims=True)  # (channels, 1)
        std = np.std(all_data, axis=1, keepdims=True)  # (channels, 1)

        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)

        logger.info("Normalization stats calculated")
        logger.info(f"Mean: {mean.squeeze()}")
        logger.info(f"Std: {std.squeeze()}")

        return {"mean": mean, "std": std}

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using calculated statistics.

        Args:
            data: Shape (time, channels, height, width)

        Returns:
            Normalized data
        """
        if self.stats is None:
            return data

        mean = self.stats["mean"][:, None, None]  # (channels, 1, 1)
        std = self.stats["std"][:, None, None]  # (channels, 1, 1)

        # Normalize
        normalized = (data - mean) / std

        return normalized

    def _augment_sequence(
        self,
        input_seq: np.ndarray,
        target_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation.

        Args:
            input_seq: (seq_len, channels, height, width)
            target_seq: (pred_horizon, channels, height, width)

        Returns:
            Augmented sequences
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            input_seq = np.flip(input_seq, axis=3).copy()
            target_seq = np.flip(target_seq, axis=3).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            input_seq = np.flip(input_seq, axis=2).copy()
            target_seq = np.flip(target_seq, axis=2).copy()

        # Random 90-degree rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            input_seq = np.rot90(input_seq, k=k, axes=(2, 3)).copy()
            target_seq = np.rot90(target_seq, k=k, axes=(2, 3)).copy()

        return input_seq, target_seq

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        import re

        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")

    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get normalization statistics."""
        return self.stats


def create_dataloaders(
    data_dir: Path,
    sequence_length: int = 10,
    prediction_horizon: int = 1,
    train_years: Optional[List[int]] = None,
    val_years: Optional[List[int]] = None,
    batch_size: int = 8,
    num_workers: int = 4,
    normalize: bool = True,
    augment_train: bool = True,
    cache_in_memory: bool = False,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory with processed features
        sequence_length: Length of input sequences
        prediction_horizon: Number of steps to predict
        train_years: Years for training (if None, auto-split)
        val_years: Years for validation (if None, auto-split)
        batch_size: Batch size
        num_workers: Number of data loading workers
        normalize: Whether to normalize
        augment_train: Whether to augment training data
        cache_in_memory: Cache data in memory

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Auto-split if years not specified
    if train_years is None or val_years is None:
        all_files = sorted(data_dir.glob("*_features*.tif"))
        all_years = sorted([
            UrbanHeatDataset._extract_year(f.name) for f in all_files
        ])

        # Use last 20% for validation
        split_idx = int(len(all_years) * 0.8)
        train_years = all_years[:split_idx]
        val_years = all_years[split_idx:]

    # Create datasets
    train_dataset = UrbanHeatDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        years=train_years,
        normalize=normalize,
        augment=augment_train,
        cache_in_memory=cache_in_memory,
    )

    val_dataset = UrbanHeatDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        years=val_years,
        normalize=normalize,
        augment=False,
        cache_in_memory=cache_in_memory,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader
  
