"""
Zero-frequency Resonator method for GCI tracking.

ZFR is basically a band-pass filter at very low frequency.
"""

import torch
from torch import Tensor

from dsp.utils import frame_signal, pad


def continuous_hann_sinc_filter(fs: int, fc: int, L: int, dtype: torch.dtype, device: torch.device):
    """Return an ideal sinc window weighted by a hann window.
    Args:
        fs: Sampling rate in integer.
        fc: Cutoff frequency in integer.
        L: window length.
    """
    assert L % 2 == 1
    assert fc < fs / 2
    hsupp = torch.linspace(-(L-1)/2, (L-1)/2, L, dtype=dtype, device=device)
    hideal = (2 * fc / fs) * torch.sinc(2 * fc * hsupp / fs)
    hann = torch.hann_window(L+1, periodic=True, dtype=dtype, device=device)[1:]
    return hideal * hann


def integrate(x: Tensor) -> Tensor:
    """Compute integration with cumsum."""
    return x.cumsum(dim=-1)


def low_pass(x: Tensor, N: int, fs: int, fc: int):
    w = continuous_hann_sinc_filter(fs, fc, 2*N+1, x.dtype, x.device)
    w = (w / w.sum()).view(1, -1)
    # [1, 2*N+1]
    framed_x = frame_signal(pad(x.view(1, 1, -1), (N, N), mode="replicate"), (2*N+1), 1)
    return torch.matmul(w, framed_x).view(-1)


def remove_moving_average_hann(x: Tensor, N: int, fs: int, fc: int):
    """Remove treend in hann moving average. Computation done in time domain.
        Args:
        x: [n_sample]
        N: the window will be [-N, N], totally 2N+1 samples.
    Returns:
        x: [n_sample]
    """
    return x - low_pass(x, N, fs, fc)


def zfr(x: Tensor, fs: int, N: int = 150, R: int = 3, fc: int = 70) -> Tensor:
    """Modified version of the ZFR filter.
    Args:
        x: signal of shape [n_sample].
        fs: sampling rate.
        N: half window size in samples.
        R: number of integrations, should be odd.
        fc: high pass cutoff frequency.
    """
    for r in range(R):
        x = remove_moving_average_hann(x, N, fs, fc)
        x = integrate(x)
        x = remove_moving_average_hann(x, N, fs, fc)
    return x
