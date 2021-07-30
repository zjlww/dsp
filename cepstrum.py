"""
Complex cepstrum utilities.
Odd length complex cepstrums are ordered like [0, 1, -1].
And even length complex cepstrums are ordered like [0, 1, -2, -1].
"""
import torch
from torch import Tensor
from .utils import reshape_zeros_like, fftpad
from typing import Tuple, Optional


def split_axis(
    x: Tensor, dim: Optional[int] = -1
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split x on dim, according to the following pattern:
        [0, 1, 2, -2, -1] => [0], [1, 2], [-2, -1], [];
        [0, 1, -2, -1] => [0], [1], [-1], [0]
    """
    size = x.size(dim)
    half = (size - 1) // 2
    pos_half = x.narrow(dim, 1, half)
    neg_half = x.narrow(dim, size - half, half)
    origin = x.narrow(dim, 0, 1)
    zero_padding = reshape_zeros_like(x, dim, size - 1 - half - half)
    return origin, pos_half, neg_half, zero_padding


def force_zero_phase(x: Tensor, dim: Optional[int] = -1) -> Tensor:
    """This transform is designed to force a complex cepstrums to have
    zero-phase response. It also forces fft time-wrapped tensors to be
    symmetric.
    The value at index x.size(dim) // 2 is zeroed out.
    Examples: see tests/dsp/test_cepstrum.py
    """
    origin, pos_half, neg_half, zero_padding = split_axis(x, dim)
    sym_pos_half = (pos_half + neg_half.flip(dim)) / 2
    sym_neg_half = (neg_half + pos_half.flip(dim)) / 2
    return torch.cat([origin, sym_pos_half, zero_padding, sym_neg_half], dim)


def force_minimum_phase(x: Tensor, dim: Optional[int] = -1) -> Tensor:
    """This transformation is designed to transform arbitrary complex cesptrums
    to its minimum-phase counter part.
    Examples: see tests/dsp/test_cepstrum.py
    """
    origin, pos_half, neg_half, zero_padding = split_axis(x, dim)
    new_pos_half = (pos_half + neg_half.flip(dim))
    new_neg_half = torch.zeros_like(pos_half)
    return torch.cat([origin, new_pos_half, zero_padding, new_neg_half], dim)


def complex_cepstrum_to_fft(
    ccep: Tensor, fft_size: int, dim: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert complex cepstrums to its corresponding fft.
    Args:
        ccep: ccep.size(dim) is ccep_size.
        fft_size: Target fft_size, should be substantially larger than
            ccep_size to avoid time wraping error.
        dim: Dimension storing the complex cepstrum. Defaults to -1.
    Returns:
        X: FFT of the impulse response x
        log(|X|): Log scale magnitude
            responses at FFT sampling points.
        arg(X): Phase responses at FFT samples.
    """
    ccep_size = ccep.size(dim)
    assert fft_size >= ccep_size, "FFT size should be greater than CCep size."
    ccep = fftpad(ccep, fft_size - ccep_size, dim=dim)
    X_hat = torch.fft.fft(ccep, dim=dim)  # [fft_size@dim]
    log_magnitude_responses = X_hat.real
    phase_responses = X_hat.imag
    magnitude_responses = log_magnitude_responses.exp()
    X_real = magnitude_responses * torch.cos(phase_responses)
    X_imag = magnitude_responses * torch.sin(phase_responses)
    X = torch.complex(X_real, X_imag)
    return X, log_magnitude_responses, phase_responses


def complex_cepstrum_to_imp(
    ccep: Tensor, fft_size: int, dim: int = -1) -> Tensor:
    """Convert complex cepstrums to corresponding magnitude responses,
    phase responses, and impulse responses.
    Args:
        ccep: [ccep_size@dim]
        fft_size: Target fft_size, should be greater than ccep_size.
    Returns:
        impulse_responses: [fft_size@dim]. Approximated time wrapped
            impulse responses.
    """
    X, _, _ = complex_cepstrum_to_fft(ccep, fft_size, dim=dim)
    x = torch.fft.ifft(X, dim=dim).real  # [fft_size@dim]
    return x


def complex_cepstrum_lowpass_mask(
    ccep_size: int, max_quefrency: int) -> Tensor:
    """Liftering mask for complex cepstrums.
    Examples:
        >>> complex_cepstrum_lowpass_mask(6, 2)
        [1, 1, 1, 0, 1, 1]
        >>> complex_cepstrum_lowpass_mask(7, 2)
        [1, 1, 1, 0, 0, 1, 1]
    """
    res = torch.ones(ccep_size)
    res[max_quefrency + 1: -max_quefrency] = 0
    return res


def complex_cepstrum_scale(
    ccep_size: int, device: Optional[torch.device] = "cpu") -> Tensor:
    """Generate 1 / |n| scale for scaling complex cepstrums.
    Returns:
        scale: shape [n].
    """
    meter = torch.linspace(-(ccep_size // 2), (ccep_size - 1) // 2,
                           ccep_size).abs()
    meter = torch.cat([meter[ccep_size // 2:], meter[: ccep_size // 2]])
    meter[0] = 1.0
    ones = torch.ones_like(meter)
    scaler = ones / meter
    return scaler.to(device)