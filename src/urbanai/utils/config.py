"""Configuration Management"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for UrbanAI."""

    def __init__(self, config: Optional[Union[str, Path, Dict]] = None) -> None:
        self.config = self._load_config(config) if config else self._default_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access."""
        return self.config[key]

    @staticmethod
    def _load_config(config: Union[str, Path, Dict]) -> Dict[str, Any]:
        """Load configuration from file or dict."""
        if isinstance(config, dict):
            return config

        config_path = Path(config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Default configuration."""
        return {
            "preprocessing": {
                "start_year": 1985,
                "end_year": 2025,
                "interval": 2,
                "season": "07-01_12-31",
                "tocantins": {
                    "enabled": False,  # Disabled by default for safety
                },
            },
            "model": {
                "architecture": "convlstm",
                "input_channels": 5,  # Will be auto-configured based on tocantins.enabled
                "hidden_dims": [64, 128, 256, 256, 128, 64],
                "kernel_size": 3,
            },
            "training": {
                "epochs": 100,
                "batch_size": 8,
                "learning_rate": 0.001,
            },
        }


def get_input_channels(config: Dict[str, Any]) -> int:
    """
    Determine the number of input channels based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        7 if Tocantins is enabled (NDBI, NDVI, NDWI, NDBSI, LST, IS, SS)
        5 if Tocantins is disabled (NDBI, NDVI, NDWI, NDBSI, LST)
    """
    tocantins_enabled = config.get("preprocessing", {}).get("tocantins", {}).get("enabled", False)
    return 7 if tocantins_enabled else 5


def get_band_names(config: Dict[str, Any]) -> list:
    """
    Get the list of band names based on configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of band names for the current configuration
    """
    base_bands = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]
    tocantins_enabled = config.get("preprocessing", {}).get("tocantins", {}).get("enabled", False)
    
    if tocantins_enabled:
        return base_bands + ["IS", "SS"]
    return base_bands
