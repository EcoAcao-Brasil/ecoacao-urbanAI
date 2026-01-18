"""
PyTorch Dataset for Spatiotemporal Urban Heat Data

Handles loading, preprocessing, and augmentation of temporal raster sequences with configurable parameters.
"""

import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import psutil
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class UrbanHeatDataset(Dataset):
    """
    PyTorch Dataset for spatiotemporal urban heat data.

    Loads temporal sequences of rasters with 7 channels:
    [NDBI, NDVI, NDWI, NDBSI, LST, IS, SS]

    Args:
        data_dir: Directory with processed feature rasters.
        sequence_length: Length of input sequences.
        prediction_horizon: Number of future steps to predict.
        years: List of years to include (if None, uses all available).
        normalize: Whether to normalize features.
        normalization_method: Method to use ('zscore', 'minmax', 'robust').
        augment: Whether to apply data augmentation.
        augment_config: Configuration dictionary for augmentation parameters.
        cache_in_memory: Cache all data in memory (faster but requires high RAM).
    """

    def __init__(
        self,
        data_dir: Path,
        sequence_length: int = 10,
        prediction_horizon: int = 1,
        years: Optional[List[int]] = None,
        normalize: bool = True,
        normalization_method: str = "zscore",
        augment: bool = False,
        augment_config: Optional[Dict] = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.augment = augment
        self.augment_config = augment_config or self._default_augment_config()
        self.cache_in_memory = cache_in_memory

        # Find available files
        self.file_paths = self._find_files(years)
        min_required = sequence_length + prediction_horizon
        if len(self.file_paths) < min_required:
            raise ValueError(
                f"Need at least {min_required} years, found {len(self.file_paths)}"
            )

        # Get spatial dimensions from first file
        with rasterio.open(self.file_paths[0]) as src:
            self.height = src.height
            self.width = src.width
            self.n_channels = src.count

        if self.n_channels != 5:
            raise ValueError(f"Expected 5 channels, found {self.n_channels}")

        # Calculate normalization statistics
        self.stats = self._calculate_stats() if normalize else None

        # Cache data if requested
        self.cache: Dict[int, np.ndarray] = {}
        if cache_in_memory:
            estimated_size = self._estimate_cache_size()
            available_memory = psutil.virtual_memory().available
            if estimated_size > available_memory * 0.7:
                logger.warning(f"Cache would use {estimated_size/1e9:.1f}GB, disabling")
                cache_in_memory = False

        logger.info(f"Dataset initialized: {len(self)} samples")
        logger.info(
            f"Seq Len: {sequence_length}, Horizon: {prediction_horizon}, "
            f"Norm: {normalization_method}, Augment: {augment}"
        )

    def __len__(self) -> int:
        """Number of available sequences."""
        return len(self.file_paths) - self.sequence_length - self.prediction_horizon + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_sequence, target_sequence)
            - input_sequence: (seq_len, channels, height, width)
            - target_sequence: (pred_horizon, channels, height, width)
        """
        # Determine indices
        input_indices = list(range(idx, idx + self.sequence_length))
        target_indices = list(
            range(
                idx + self.sequence_length,
                idx + self.sequence_length + self.prediction_horizon,
            )
        )

        # Load data (from cache or disk)
        if self.cache_in_memory:
            input_data = [self.cache[i] for i in input_indices]
            target_data = [self.cache[i] for i in target_indices]
        else:
            input_data = [self._load_raster(self.file_paths[i]) for i in input_indices]
            target_data = [self._load_raster(self.file_paths[i]) for i in target_indices]

        # Stack into sequences
        input_seq = np.stack(input_data, axis=0)  # (seq_len, C, H, W)
        target_seq = np.stack(target_data, axis=0)  # (pred_horizon, C, H, W)

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
        # Try primary naming pattern, then fallback
        files = sorted(self.data_dir.glob("*_features_complete.tif"))
        if not files:
            files = sorted(self.data_dir.glob("*_features.tif"))

        if not files:
            raise ValueError(f"No feature files found in {self.data_dir}")

        # Filter by years if specified
        if years is not None:
            files = [f for f in files if self._extract_year(f.name) in years]

        # Sort by year
        files = sorted(files, key=lambda f: self._extract_year(f.name))
        return files

    def _load_raster(self, path: Path) -> np.ndarray:
        """
        Load raster file.

        Returns:
            Array of shape (channels, height, width), float32.
        """
        with rasterio.open(path) as src:
            data = src.read()

        # Handle NaN/Inf
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data.astype(np.float32)

    def _estimate_cache_size(self) -> int:
        """
        Estimate memory required for caching all data in bytes.
        
        Returns:
            Estimated size in bytes
        """
        # Rough estimate: n_files * channels * height * width * 4 bytes (float32)
        size_per_file = self.n_channels * self.height * self.width * 4
        total_size = len(self.file_paths) * size_per_file
        
        logger.debug(f"Estimated cache size: {total_size / 1e9:.2f} GB")
        return total_size
    
    def _calculate_stats(self) -> Dict[str, np.ndarray]:
        """Calculate statistics for the selected normalization method."""
        logger.info(f"Calculating {self.normalization_method} statistics...")

        all_data = []
        for path in self.file_paths:
            data = self._load_raster(path)
            # Reshape to (channels, -1)
            all_data.append(data.reshape(data.shape[0], -1))

        # Concatenate all pixels
        all_data = np.concatenate(all_data, axis=1)

        stats = {}
        if self.normalization_method == "zscore":
            mean = np.mean(all_data, axis=1, keepdims=True)
            std = np.std(all_data, axis=1, keepdims=True)
            stats = {"mean": mean, "std": np.where(std < 1e-8, 1.0, std)}

        elif self.normalization_method == "minmax":
            min_vals = np.min(all_data, axis=1, keepdims=True)
            max_vals = np.max(all_data, axis=1, keepdims=True)
            rng = max_vals - min_vals
            stats = {"min": min_vals, "range": np.where(rng < 1e-8, 1.0, rng)}

        elif self.normalization_method == "robust":
            median = np.median(all_data, axis=1, keepdims=True)
            q75 = np.percentile(all_data, 75, axis=1, keepdims=True)
            q25 = np.percentile(all_data, 25, axis=1, keepdims=True)
            iqr = q75 - q25
            stats = {"median": median, "iqr": np.where(iqr < 1e-8, 1.0, iqr)}

        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")

        logger.info("Normalization statistics calculated.")
        return stats

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization using pre-calculated statistics."""
        if self.stats is None:
            return data

        if self.normalization_method == "zscore":
            return (data - self.stats["mean"][:, None, None]) / self.stats["std"][:, None, None]

        if self.normalization_method == "minmax":
            return (data - self.stats["min"][:, None, None]) / self.stats["range"][:, None, None]

        if self.normalization_method == "robust":
            return (data - self.stats["median"][:, None, None]) / self.stats["iqr"][:, None, None]

        return data

    def _augment_sequence(
        self, input_seq: np.ndarray, target_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation based on configuration."""
        cfg = self.augment_config

        # Horizontal flip
        if np.random.rand() < cfg["horizontal_flip_prob"]:
            input_seq = np.flip(input_seq, axis=3).copy()
            target_seq = np.flip(target_seq, axis=3).copy()

        # Vertical flip
        if np.random.rand() < cfg["vertical_flip_prob"]:
            input_seq = np.flip(input_seq, axis=2).copy()
            target_seq = np.flip(target_seq, axis=2).copy()

        # Random rotation
        if np.random.rand() < cfg["rotation_prob"]:
            angle = np.random.choice(cfg["rotation_angles"])
            k = angle // 90
            input_seq = np.rot90(input_seq, k=k, axes=(2, 3)).copy()
            target_seq = np.rot90(target_seq, k=k, axes=(2, 3)).copy()

        # Gaussian noise
        if np.random.rand() < cfg["noise_prob"]:
            noise_std = cfg["noise_std"]
            input_seq += np.random.normal(0, noise_std, input_seq.shape).astype(np.float32)
            target_seq += np.random.normal(0, noise_std, target_seq.shape).astype(np.float32)

        return input_seq, target_seq

    @staticmethod
    def _default_augment_config() -> Dict:
        """Default augmentation parameters."""
        return {
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "rotation_prob": 0.5,
            "rotation_angles": [90, 180, 270],
            "noise_prob": 0.0,
            "noise_std": 0.01,
        }

    @staticmethod
    def _extract_year(filename: str) -> int:
        """Extract year from filename."""
        match = re.search(r"(\d{4})", filename)
        if match:
            return int(match.group(1))
        raise ValueError(f"Could not extract year from: {filename}")

    def get_normalization_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get normalization statistics."""
        return self.stats


def create_dataloaders(
    data_dir: Path,
    config: Dict,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders using full configuration.

    Args:
        data_dir: Directory with processed features.
        config: Complete configuration dictionary.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_cfg = config.get("training", {})
    adv_cfg = config.get("advanced", {})
    
    # Dataset Parameters
    seq_len = train_cfg.get("sequence_length", 10)
    pred_horizon = train_cfg.get("prediction_horizon", 1)
    norm_method = train_cfg.get("normalization_method", "zscore")
    normalize = train_cfg.get("normalize", True)
    
    # Loader Parameters
    batch_size = train_cfg.get("batch_size", 8)
    num_workers = train_cfg.get("num_workers", 4)
    cache_mem = train_cfg.get("cache_in_memory", False)
    pin_mem = train_cfg.get("pin_memory", True) and torch.cuda.is_available()

    # Split Strategy
    split_cfg = train_cfg.get("train_val_split", {})
    split_method = split_cfg.get("method", "temporal")
    
    all_files = sorted(data_dir.glob("*_features*.tif"))
    all_years = sorted([UrbanHeatDataset._extract_year(f.name) for f in all_files])

    if split_method == "manual":
        train_years = split_cfg.get("train_years")
        val_years = split_cfg.get("val_years")
        if train_years is None or val_years is None:
            raise ValueError("Manual split requires 'train_years' and 'val_years'")
            
    elif split_method == "random":
        random.seed(adv_cfg.get("random_seed", 42))
        years_shuffled = list(all_years)
        random.shuffle(years_shuffled)
        
        split_idx = int(len(years_shuffled) * split_cfg.get("train_ratio", 0.8))
        train_years = sorted(years_shuffled[:split_idx])
        val_years = sorted(years_shuffled[split_idx:])
        
    else:  # "temporal" default
        split_idx = int(len(all_years) * split_cfg.get("train_ratio", 0.8))
        train_years = all_years[:split_idx]
        val_years = all_years[split_idx:]

    logger.info(f"Split Strategy: {split_method}")
    logger.info(f"Train Years ({len(train_years)}): {train_years}")
    logger.info(f"Val Years ({len(val_years)}): {val_years}")

    # Create Datasets
    aug_cfg = train_cfg.get("augmentation", {})
    should_augment = aug_cfg.get("enabled", True)

    train_dataset = UrbanHeatDataset(
        data_dir=data_dir,
        sequence_length=seq_len,
        prediction_horizon=pred_horizon,
        years=train_years,
        normalize=normalize,
        normalization_method=norm_method,
        augment=should_augment,
        augment_config=aug_cfg,
        cache_in_memory=cache_mem,
    )

    val_dataset = UrbanHeatDataset(
        data_dir=data_dir,
        sequence_length=seq_len,
        prediction_horizon=pred_horizon,
        years=val_years,
        normalize=normalize,
        normalization_method=norm_method,
        augment=False,  # Never augment validation
        cache_in_memory=cache_mem,
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=False,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader
