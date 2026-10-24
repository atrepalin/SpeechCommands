import torch
import torch.nn as nn

from src.model.se.seblock1d import SEBlock1D


class ResBlock1D(nn.Module):
    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        dropout: float = 0.0,
        reduction: int = 8,
    ):
        super().__init__()

        kernel_size = 5

        # Calculate padding to obtain same output shape
        padding = ((kernel_size - 1) // 2) * dilation

        self.bn1 = nn.BatchNorm1d(channels)
        self.act1 = nn.LeakyReLU()

        # For audio 1d processing we use dilated convolutions to capture long-range dependencies
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.bn2 = nn.BatchNorm1d(channels)
        self.act2 = nn.LeakyReLU()

        self.dropout = nn.Dropout1d(dropout)

        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

        self.se = SEBlock1D(channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = self.se(x)

        return x + residual
