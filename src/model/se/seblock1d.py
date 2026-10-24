import torch
import torch.nn as nn


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        hidden = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        b, c, _ = x.shape

        s = self.pool(x).view(b, c)
        s = self.fc(s)
        s = s.view(b, c, 1)

        return x * s
