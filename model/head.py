import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)
