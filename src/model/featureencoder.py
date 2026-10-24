from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FeatureEncoderLayer:
    out_channels: int
    kernel_size: int
    stride: int


class FeatureEncoder(nn.Module):
    def __init__(self, configs: list[FeatureEncoderLayer]):
        super().__init__()

        layers = []

        in_channels = 1

        for layer in configs:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    layer.out_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    bias=False,
                )
            )

            layers.append(nn.LeakyReLU())

            in_channels = layer.out_channels

        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(x)
