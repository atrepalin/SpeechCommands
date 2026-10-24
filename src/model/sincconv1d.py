import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1d(nn.Module):
    def __init__(
        self,
        out_channels=80,
        kernel_size=251,
        sample_rate=16000,
        min_low_hz=50.0,
        min_band_hz=50.0,
        stride=10,
        normalize=True,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.stride = stride
        self.normalize = normalize

        # maximum nyquist frequency
        self.nyquist = sample_rate / 2.0

        # use mel scale for initial values
        mel_edges = torch.linspace(
            self._hz_to_mel(min_low_hz),
            self._hz_to_mel(self.nyquist - 1.0),
            out_channels + 1,
        )

        edges = self._mel_to_hz(mel_edges)

        initial_low = edges[:-1]
        initial_high = edges[1:]

        # Normalized initial values
        low_min = min_low_hz
        low_max = self.nyquist - min_band_hz

        low_ratio = (initial_low - low_min) / (low_max - low_min)

        low_ratio = low_ratio.clamp(1e-4, 1.0 - 1e-4)

        # Use logit parameterization
        self.low_param = nn.Parameter(torch.log(low_ratio / (1.0 - low_ratio)))

        # Band ratio
        available_band = self.nyquist - initial_low - min_band_hz

        desired_band = initial_high - initial_low - min_band_hz

        band_ratio = desired_band / available_band.clamp_min(1.0)

        band_ratio = band_ratio.clamp(1e-4, 1.0 - 1e-4)

        # Use logit parameterization
        self.band_param = nn.Parameter(torch.log(band_ratio / (1.0 - band_ratio)))

        # Symmetric time axis
        n = torch.arange(
            -(kernel_size // 2),
            kernel_size // 2 + 1,
            dtype=torch.float32,
        )

        self.register_buffer("n", n)

        # Use hamming window for better frequency resolution
        self.register_buffer(
            "window",
            torch.hamming_window(
                kernel_size,
                periodic=False,
            ),
        )

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        """Converts a frequency in hertz to a mel value"""
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        """Converts a mel value to a frequency in hertz"""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _cutoff_frequencies(self) -> tuple[torch.Tensor, torch.Tensor]:
        low_max = self.nyquist - self.min_band_hz

        # Calculate low frequency from parameter
        low = self.min_low_hz + (low_max - self.min_low_hz) * torch.sigmoid(
            self.low_param
        )

        max_band = self.nyquist - low - self.min_band_hz

        # Calculate band from parameter
        band = self.min_band_hz + max_band * torch.sigmoid(self.band_param)

        high = low + band

        # Ensure high frequency does not exceed nyquist
        high = torch.minimum(high, torch.full_like(high, self.nyquist - 1e-3))

        return low, high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, 1, T], got {x.shape}")

        if x.shape[1] != 1:
            raise ValueError("SincConv1d expects mono input.")

        dtype = x.dtype
        device = x.device

        low, high = self._cutoff_frequencies()

        low = low.to(device=device, dtype=dtype)
        high = high.to(device=device, dtype=dtype)

        n = self.n.to(device=device, dtype=dtype)

        window = self.window.to(device=device, dtype=dtype)

        # [C, 1]
        low = low[:, None]
        high = high[:, None]

        t = n[None, :]

        low_norm = low / self.sample_rate
        high_norm = high / self.sample_rate

        # Sinc band pass filter
        filters = 2.0 * high_norm * torch.sinc(
            2.0 * high_norm * t
        ) - 2.0 * low_norm * torch.sinc(2.0 * low_norm * t)

        # Apply hamming window
        filters = filters * window[None, :]

        if self.normalize:
            # L1 normalization
            norm = (
                filters.abs()
                .sum(
                    dim=1,
                    keepdim=True,
                )
                .clamp_min(1e-8)
            )

            filters = filters / norm

        # [C, 1, K]
        filters = filters[:, None, :]

        # Convolution with filters
        return F.conv1d(x, filters, stride=self.stride)

    def extra_repr(self):
        return (
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"sample_rate={self.sample_rate}, "
            f"min_low_hz={self.min_low_hz}, "
            f"min_band_hz={self.min_band_hz}, "
            f"stride={self.stride}, "
            f"normalize={self.normalize}"
        )
