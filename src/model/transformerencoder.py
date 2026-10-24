import torch
import torch.nn as nn

from src.model.transformerblock import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(self, dim=384, num_layers=6, num_heads=6, ff_dim=1536, dropout=0.1):
        super().__init__()

        # Transformer based encoder
        self.layers = nn.Sequential(
            *[
                TransformerBlock(
                    dim=dim, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout
                )
                for _ in range(num_layers)
            ],
            nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
