"""GPU and Device Management"""

import torch
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Requested device ('cuda', 'cpu', 'auto', or None)

    Returns:
        torch.device
    """
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def log_gpu_info() -> None:
    """Output GPU information."""
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: True")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            logger.info(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        logger.info("CUDA Available: False")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
