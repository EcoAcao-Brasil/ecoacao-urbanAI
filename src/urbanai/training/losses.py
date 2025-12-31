"""Custom Loss Functions"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialMSELoss(nn.Module):
    """
    Spatially-weighted MSE loss.

    Gives higher weight to urban core pixels and anomaly zones.
    """

    def __init__(self, core_weight: float = 2.0) -> None:
        super().__init__()
        self.core_weight = core_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate weighted MSE loss.

        Args:
            predictions: (batch, time, channels, h, w)
            targets: (batch, time, channels, h, w)
            weights: Optional spatial weights (batch, 1, 1, h, w)

        Returns:
            Scalar loss
        """
        # Basic MSE
        mse = F.mse_loss(predictions, targets, reduction="none")

        # Apply spatial weights if provided
        if weights is not None:
            mse = mse * weights

        # Weight LST channel more (channel 4)
        channel_weights = torch.ones(predictions.shape[2], device=predictions.device)
        channel_weights[4] = 2.0  # LST
        channel_weights[5] = 1.5  # IS
        channel_weights[6] = 1.5  # SS

        # Reshape for broadcasting
        channel_weights = channel_weights.view(1, 1, -1, 1, 1)
        mse = mse * channel_weights

        return mse.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using spatial features.

    Compares high-level spatial patterns rather than pixel-wise differences.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate perceptual loss.

        Args:
            predictions: (batch, time, channels, h, w)
            targets: (batch, time, channels, h, w)

        Returns:
            Scalar loss
        """
        # Calculate spatial gradients
        pred_grad_x = predictions[:, :, :, :, 1:] - predictions[:, :, :, :, :-1]
        pred_grad_y = predictions[:, :, :, 1:, :] - predictions[:, :, :, :-1, :]

        target_grad_x = targets[:, :, :, :, 1:] - targets[:, :, :, :, :-1]
        target_grad_y = targets[:, :, :, 1:, :] - targets[:, :, :, :-1, :]

        # Gradient loss
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class CombinedLoss(nn.Module):
    """
    Combined loss: MSE + Perceptual.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        perceptual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        self.mse_loss = SpatialMSELoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined loss."""
        mse = self.mse_loss(predictions, targets)
        perceptual = self.perceptual_loss(predictions, targets)

        total_loss = self.mse_weight * mse + self.perceptual_weight * perceptual

        return total_loss
