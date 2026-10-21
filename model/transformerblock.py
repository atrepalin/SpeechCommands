import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim=384, num_heads=6, ff_dim=1536, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm1(x)

        attn_out, _ = self.attention(x, x, x)

        x = residual + self.dropout1(attn_out)

        residual = x

        x = self.norm2(x)
        x = self.ffn(x)

        x = residual + self.dropout2(x)

        return x
