"""
Loss Functions

Defines the objective functions used to optimize the UrbanAI models.
Currently focuses on regression metrics for heat map prediction.
"""

import torch
import torch.nn as nn


class SimpleMSELoss(nn.Module):
    """
    Wrapper around PyTorch's standard Mean Squared Error loss.
    
    While simple, this class structure allows us to easily swap in more complex 
    spatial loss functions (like SSIM or Perceptual Loss) later without 
    refactoring the training loop.
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
        Computes element-wise MSE between predicted heat maps and ground truth.
        
        Expects tensors of shape (batch, time, channels, h, w).
        """
        return self.mse(predictions, targets)
