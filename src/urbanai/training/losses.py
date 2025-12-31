"""
Loss Functions

Defines the objective functions used to optimize the UrbanAI models.
"""

import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Mean Squared Error loss for spatiotemporal prediction.
    
    Simple wrapper that allows easy extension to more complex losses later.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute MSE between predictions and targets.
        
        Args:
            predictions: (batch, time, channels, h, w)
            targets: (batch, time, channels, h, w)
            
        Returns:
            Scalar loss value
        """
        return self.mse(predictions, targets)