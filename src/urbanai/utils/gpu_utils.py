"""GPU and Device Management"""

import torch
from typing import Optional


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


def print_gpu_info() -> None:
    """Print GPU information."""
    if torch.cuda.is_available():
        print(f"CUDA Available: True")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA Available: False")


def clear_gpu_memory() -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
