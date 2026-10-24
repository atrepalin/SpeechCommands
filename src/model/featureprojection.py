import torch
import torch.nn as nn


class FeatureProjection(nn.Module):
    def __init__(self, input_dim: int = 512, hidden_dim: int = 384, dropout: int = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)

        return x
