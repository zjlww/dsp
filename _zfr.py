"""
Zero-frequency Resonator method for GCI tracking.
ZFR is basically a band-pass filter at very low frequency.
"""

import torch
from torch import Tensor


def continuous_hann_sinc_filter(
    fs: int, fc: float, L: int, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """Return an ideal sinc window weighted by a hann window.
    This will create a nearly ideal zero-phase low-pass filter.
    Args:
        fs: Sampling rate in integer.
        fc: Cutoff frequency in integer.
        L: window length.
    Returns:
        filter: FloatTensor [L].
    """
    assert L % 2 == 1
    assert fc < fs / 2
    hsupp = torch.linspace(-(L-1)/2, (L-1)/2, L, dtype=dtype, device=device)
    hideal = (2 * fc / fs) * torch.sinc(2 * fc * hsupp / fs)
    hann = torch.hann_window(L, dtype=dtype, device=device)
    return hideal * hann


def hann_sinc_low_pass(x: Tensor, N: int, fs: int, fc: float) -> Tensor:
    """Hann windowed ideal low pass filter.
    Args:
        x: [n_batch, 1, n_sample]
        N: the window will be [-N, N], totally 2N+1 samples.
    Returns:
        y: [n_batch, 1, n_sample]
    """
    w = continuous_hann_sinc_filter(fs, fc, 2*N+1, x.dtype, x.device)
    w = (w / w.sum()).view(1, 1, -1)
    return torch.nn.functional.conv1d(x, w, padding=N)


def hann_sinc_high_pass(x: Tensor, N: int, fs: int, fc: float) -> Tensor:
    """High-pass hann windowed ideal filter. See `hann_sinc_low_pass`
    for details."""
    return x - hann_sinc_low_pass(x, N, fs, fc)


def zfr(x: Tensor, fs: int, N: int = 150, R: int = 3, fc: float = 70.) -> Tensor:
    """Modified version of the ZFR filter.
    Args:
        x: [n_batch, 1, n_sample].
        fs: sampling rate.
        N: half window size in samples.
        R: number of integrations, should be odd.
        fc: high pass cutoff frequency.
    Returns:
        y: ZFR filter output, [n_batch, 1, n_sample].
    """
    for _ in range(R):
        x = hann_sinc_high_pass(x, N, fs, fc)
        x = x.cumsum(dim=-1)
        x = hann_sinc_high_pass(x, N, fs, fc)
    return x
