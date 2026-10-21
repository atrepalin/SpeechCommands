import torch
import torch.nn as nn


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int, attention_channels: int | None = None):
        super().__init__()

        if attention_channels is None:
            attention_channels = channels // 2

        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.attention(x)
        weights = torch.softmax(scores, dim=-1)

        pooled = torch.sum(x * weights, dim=-1)

        return pooled
