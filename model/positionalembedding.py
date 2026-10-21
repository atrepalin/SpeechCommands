import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, dim=384, kernel_size=128, groups=16):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=0, groups=groups
        )

        self.activation = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)

        x = x.transpose(1, 2)

        left = self.kernel_size // 2
        right = self.kernel_size - 1 - left

        x = F.pad(x, (left, right))

        x = self.conv(x)

        x = self.activation(x)

        x = x.transpose(1, 2)

        x = x[:, :T, :]

        return x
