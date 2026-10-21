import torch
import torch.nn as nn

from model.se.seblock2d import SEBlock2D


class ResBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.05,
        reduction: int = 8,
    ):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.LeakyReLU()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.LeakyReLU()

        self.dropout = nn.Dropout2d(dropout)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.se = SEBlock2D(out_channels, reduction=reduction)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.conv2(x)

        x = self.se(x)

        return x + residual
