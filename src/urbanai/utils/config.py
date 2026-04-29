"""Configuration utilities"""

from typing import Any, Dict, List


def get_input_channels(config: Dict[str, Any]) -> int:
    """
    Determine the number of input channels based on configuration.

    Returns 7 if Tocantins is enabled (NDBI, NDVI, NDWI, NDBSI, LST, IS, SS),
    or 5 if disabled (NDBI, NDVI, NDWI, NDBSI, LST).
    """
    tocantins_enabled = config.get("preprocessing", {}).get("tocantins", {}).get("enabled", False)
    return 7 if tocantins_enabled else 5


def get_band_names(config: Dict[str, Any]) -> List[str]:
    """
    Get the list of band names based on configuration.

    Returns 7 band names if Tocantins is enabled, 5 otherwise.
    """
    base_bands = ["NDBI", "NDVI", "NDWI", "NDBSI", "LST"]
    tocantins_enabled = config.get("preprocessing", {}).get("tocantins", {}).get("enabled", False)
    if tocantins_enabled:
        return base_bands + ["IS", "SS"]
    return base_bands
