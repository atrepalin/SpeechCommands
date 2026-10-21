from collections import OrderedDict

import torch
import torch.nn as nn
import torchaudio

from model.attentionpooling1d import AttentionPooling1D
from model.downsample.downsample2d import Downsample2D
from model.head import ClassificationHead
from model.res.resblock2d import ResBlock2D


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)

        return (x - mean) / (std + self.eps)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


class InputPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)

        return x


class MelResNetKWS(nn.Module):
    def __init__(
        self,
        num_classes: int,
    ):
        super().__init__()

        self.backbone = nn.Sequential(OrderedDict([
            ("preprocess", InputPreprocess()),
            (
                "mel",
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=400,
                    win_length=400,
                    hop_length=160,
                    n_mels=64,
                    f_min=20,
                    f_max=16000 // 2,
                    power=2.0,
                ),
            ),
            ("db", torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)),
            ("norm", SpecNormalize()),
            ("unsqueeze", Unsqueeze(1)),
            (
                "stem",
                nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size=5, padding=2, bias=False),
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(),
                ),
            ),
            (
                "stage1",
                nn.Sequential(
                    ResBlock2D(64, 64, dropout=0.05),
                    ResBlock2D(64, 64, dropout=0.05),
                ),
            ),
            ("down1", Downsample2D(64, 128, kernel_size=3, stride=2, padding=1)),
            (
                "stage2",
                nn.Sequential(
                    ResBlock2D(128, 128, dropout=0.05),
                    ResBlock2D(128, 128, dropout=0.05),
                ),
            ),
            ("down2", Downsample2D(128, 256, kernel_size=3, stride=2, padding=1)),
            (
                "stage3",
                nn.Sequential(
                    ResBlock2D(256, 256, dropout=0.05),
                    ResBlock2D(256, 256, dropout=0.05),
                ),
            ),
            ("freq_pool", nn.AdaptiveAvgPool2d((1, None))),
            ("squeeze", Squeeze(2)),
            ("pool", AttentionPooling1D(channels=256, attention_channels=128)),
        ]))

        self.head = ClassificationHead(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.head(x)
        return x