from collections import OrderedDict

import torch
import torch.nn as nn

from src.model.attentionpooling1d import AttentionPooling1D
from src.model.downsample.downsample1d import Downsample1D
from src.model.head import ClassificationHead
from src.model.res.resblock1d import ResBlock1D
from src.model.sincconv1d import SincConv1d


class SincResNetKWS(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            OrderedDict(
                [
                    (
                        "sincconv",
                        SincConv1d(
                            out_channels=80,
                            kernel_size=251,
                            sample_rate=16000,
                            min_low_hz=50.0,
                            min_band_hz=50.0,
                            stride=10,
                            normalize=False,
                        ),
                    ),  # Sinc based frontend
                    (
                        "stem",
                        nn.Sequential(
                            nn.Conv1d(80, 128, kernel_size=5, padding=2, bias=False),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(),
                        ),
                    ),  # Initial feature extraction
                    (
                        "stage1",
                        nn.Sequential(
                            ResBlock1D(128, dilation=1, dropout=0.05),
                            ResBlock1D(128, dilation=2, dropout=0.05),
                            ResBlock1D(128, dilation=4, dropout=0.05),
                            ResBlock1D(128, dilation=8, dropout=0.05),
                        ),
                    ),
                    (
                        "down1",
                        Downsample1D(128, 192, kernel_size=5, stride=2, padding=2),
                    ),
                    (
                        "stage2",
                        nn.Sequential(
                            ResBlock1D(192, dilation=1, dropout=0.05),
                            ResBlock1D(192, dilation=2, dropout=0.05),
                            ResBlock1D(192, dilation=4, dropout=0.05),
                            ResBlock1D(192, dilation=8, dropout=0.05),
                        ),
                    ),
                    (
                        "down2",
                        Downsample1D(192, 256, kernel_size=5, stride=2, padding=2),
                    ),
                    (
                        "stage3",
                        nn.Sequential(
                            ResBlock1D(256, dilation=1, dropout=0.05),
                            ResBlock1D(256, dilation=2, dropout=0.05),
                            ResBlock1D(256, dilation=4, dropout=0.05),
                        ),
                    ),
                    (
                        "pool",
                        AttentionPooling1D(channels=256, attention_channels=128),
                    ),  # Attention pooling
                ]
            )
        )

        self.head = ClassificationHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x
