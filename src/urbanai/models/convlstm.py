"""
ConvLSTM Model Implementation

Based on: Shi et al. (2015) - Convolutional LSTM Network
Adapted for spatiotemporal urban heat prediction.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell.

    Args:
        input_dim: Number of input channels
        hidden_dim: Number of hidden channels
        kernel_size: Size of convolutional kernel
        bias: Whether to use bias
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: Tuple[int, int],
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.bias = bias

        # Gates: input, forget, cell, output
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        cur_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_tensor: (batch, channels, height, width)
            cur_state: (h, c) hidden and cell states

        Returns:
            Next hidden state and cell state
        """
        h_cur, c_cur = cur_state

        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # Convolve
        combined_conv = self.conv(combined)

        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        # Apply activations
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)  # Cell gate

        # Update cell state
        c_next = f * c_cur + i * g

        # Update hidden state
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size: int, spatial_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden and cell states.

        Args:
            batch_size: Batch size
            spatial_size: (height, width) of feature maps

        Returns:
            Initialized (h, c) states
        """
        height, width = spatial_size
        device = self.conv.weight.device

        h = torch.zeros(
            batch_size,
            self.hidden_dim,
            height,
            width,
            dtype=torch.float32,
            device=device,
        )
        c = torch.zeros(
            batch_size,
            self.hidden_dim,
            height,
            width,
            dtype=torch.float32,
            device=device,
        )

        return h, c


class ConvLSTM(nn.Module):
    """
    Stacked Convolutional LSTM layers.

    Args:
        input_dim: Number of input channels
        hidden_dims: List of hidden dimensions for each layer
        kernel_size: Size of convolutional kernels
        num_layers: Number of ConvLSTM layers
        batch_first: If True, input is (batch, time, channels, height, width)
        bias: Whether to use bias
        return_all_layers: Return outputs from all layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: Tuple[int, int] = (3, 3),
        num_layers: int = 1,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        super().__init__()

        # Validate inputs
        if len(hidden_dims) != num_layers:
            raise ValueError(f"Length of hidden_dims ({len(hidden_dims)}) must match num_layers ({num_layers})")

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        # Create ConvLSTM cells
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dims[i - 1]
            cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dims[i],
                    kernel_size=self.kernel_size[i],
                    bias=bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through ConvLSTM.

        Args:
            input_tensor: (batch, time, channels, height, width) or
                         (time, batch, channels, height, width)
            hidden_state: Initial hidden states for each layer

        Returns:
            layer_output_list: Outputs from each layer
            last_state_list: Final (h, c) states for each layer
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, spatial_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=(h, c),
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(
        self,
        batch_size: int,
        spatial_size: Tuple[int, int],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden states for all layers."""
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, spatial_size))
        return init_states

    @staticmethod
    def _extend_for_multilayer(
        param: Tuple[int, int],
        num_layers: int,
    ) -> List[Tuple[int, int]]:
        """Extend parameter for multiple layers."""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMEncoderDecoder(nn.Module):
    """
    ConvLSTM Encoder-Decoder for spatiotemporal prediction.

    Encoder processes input sequence, decoder generates future predictions.

    Args:
        input_channels: Number of input feature channels
        hidden_dims: Hidden dimensions for encoder layers
        kernel_size: Convolutional kernel size
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        output_channels: Number of output channels
        bias: Use bias in convolutions
    """

    def __init__(
        self,
        input_channels: int = 5,
        hidden_dims: List[int] = [64, 128, 256],
        kernel_size: Tuple[int, int] = (3, 3),
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        output_channels: int = 5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Encoder
        self.encoder = ConvLSTM(
            input_dim=input_channels,
            hidden_dims=hidden_dims[:num_encoder_layers],
            kernel_size=kernel_size,
            num_layers=num_encoder_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=False,
        )

        # Decoder
        decoder_hidden_dims = hidden_dims[num_encoder_layers:]
        if not decoder_hidden_dims:
            decoder_hidden_dims = list(reversed(hidden_dims[:num_encoder_layers]))

        self.decoder = ConvLSTM(
            input_dim=hidden_dims[num_encoder_layers - 1],
            hidden_dims=decoder_hidden_dims,
            kernel_size=kernel_size,
            num_layers=num_decoder_layers,
            batch_first=True,
            bias=bias,
            return_all_layers=False,
        )

        # Output projection
        self.output_conv = nn.Conv2d(
            in_channels=decoder_hidden_dims[-1],
            out_channels=output_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
        future_steps: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, time, channels, height, width)
            future_steps: Number of future time steps to predict

        Returns:
            Predictions (batch, future_steps, channels, height, width)
        """
        # Encode
        encoder_output, encoder_state = self.encoder(x)

        # Use final encoder hidden state as decoder initial state
        # Get the last output from encoder
        last_encoder_output = encoder_output[0][:, -1:, :, :, :]  # (batch, 1, hidden, h, w)

        # Decode
        decoder_input = last_encoder_output
        outputs = []

        for _ in range(future_steps):
            decoder_output, decoder_state = self.decoder(decoder_input)
            # Get last time step
            out = decoder_output[0][:, -1, :, :, :]  # (batch, hidden, h, w)

            # Project to output space
            prediction = self.output_conv(out)  # (batch, output_channels, h, w)
            outputs.append(prediction)

            # Use prediction as next input (with channel adjustment)
            decoder_input = decoder_output[0][:, -1:, :, :, :]

        # Stack predictions
        predictions = torch.stack(outputs, dim=1)  # (batch, future_steps, channels, h, w)

        return predictions
