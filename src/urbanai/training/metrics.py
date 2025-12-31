"""Evaluation Metrics"""

import torch
from typing import Dict


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate evaluation metrics.

    Args:
        predictions: (batch, time, channels, h, w)
        targets: (batch, time, channels, h, w)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Mean Absolute Error
    mae = torch.mean(torch.abs(predictions - targets)).item()
    metrics["mae"] = mae

    # Root Mean Squared Error
    mse = torch.mean((predictions - targets) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    metrics["rmse"] = rmse

    # R² Score (per channel)
    for i, channel in enumerate(["NDBI", "NDVI", "NDWI", "NDBSI", "LST", "IS", "SS"]):
        pred_channel = predictions[:, :, i, :, :]
        target_channel = targets[:, :, i, :, :]

        # Calculate R²
        ss_res = torch.sum((target_channel - pred_channel) ** 2)
        ss_tot = torch.sum((target_channel - target_channel.mean()) ** 2)

        if ss_tot > 0:
            r2 = 1 - (ss_res / ss_tot)
            metrics[f"r2_{channel}"] = r2.item()

    # Spatial correlation (per timestep)
    for t in range(predictions.shape[1]):
        pred_t = predictions[:, t, :, :, :].flatten()
        target_t = targets[:, t, :, :, :].flatten()

        # Pearson correlation
        pred_mean = pred_t.mean()
        target_mean = target_t.mean()

        numerator = torch.sum((pred_t - pred_mean) * (target_t - target_mean))
        denominator = torch.sqrt(
            torch.sum((pred_t - pred_mean) ** 2) * torch.sum((target_t - target_mean) ** 2)
        )

        if denominator > 0:
            correlation = (numerator / denominator).item()
            metrics[f"correlation_t{t}"] = correlation

    # Mean correlation
    corr_keys = [k for k in metrics.keys() if k.startswith("correlation_")]
    if corr_keys:
        metrics["correlation_mean"] = sum(metrics[k] for k in corr_keys) / len(corr_keys)

    return metrics


def calculate_channel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    channel_idx: int,
    channel_name: str,
) -> Dict[str, float]:
    """
    Calculate metrics for specific channel.

    Args:
        predictions: (batch, time, channels, h, w)
        targets: (batch, time, channels, h, w)
        channel_idx: Channel index
        channel_name: Channel name

    Returns:
        Dictionary of channel-specific metrics
    """
    pred_channel = predictions[:, :, channel_idx, :, :]
    target_channel = targets[:, :, channel_idx, :, :]

    metrics = {}

    # MAE
    metrics[f"{channel_name}_mae"] = torch.mean(torch.abs(pred_channel - target_channel)).item()

    # RMSE
    mse = torch.mean((pred_channel - target_channel) ** 2).item()
    metrics[f"{channel_name}_rmse"] = torch.sqrt(torch.tensor(mse)).item()

    # Bias
    bias = torch.mean(pred_channel - target_channel).item()
    metrics[f"{channel_name}_bias"] = bias

    # Range comparison
    pred_range = pred_channel.max() - pred_channel.min()
    target_range = target_channel.max() - target_channel.min()
    metrics[f"{channel_name}_range_ratio"] = (pred_range / target_range).item()

    return metrics


def calculate_spatial_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """
    Calculate spatial pattern metrics.

    Args:
        predictions: (batch, time, channels, h, w)
        targets: (batch, time, channels, h, w)

    Returns:
        Dictionary of spatial metrics
    """
    metrics = {}

    # Spatial gradients
    pred_grad_x = predictions[:, :, :, :, 1:] - predictions[:, :, :, :, :-1]
    pred_grad_y = predictions[:, :, :, 1:, :] - predictions[:, :, :, :-1, :]

    target_grad_x = targets[:, :, :, :, 1:] - targets[:, :, :, :, :-1]
    target_grad_y = targets[:, :, :, 1:, :] - targets[:, :, :, :-1, :]

    # Gradient similarity
    metrics["gradient_x_mae"] = torch.mean(torch.abs(pred_grad_x - target_grad_x)).item()
    metrics["gradient_y_mae"] = torch.mean(torch.abs(pred_grad_y - target_grad_y)).item()

    # Variance similarity
    pred_var = torch.var(predictions, dim=[3, 4]).mean().item()
    target_var = torch.var(targets, dim=[3, 4]).mean().item()
    metrics["variance_ratio"] = pred_var / (target_var + 1e-8)

    return metrics
