import torch
from torch import Tensor

from math import pi
from typing import Union, Optional


def freq_multiplier(n_harmonic: int, device: torch.device) -> Tensor:
    """Generate the frequency multiplier [[[1], [2], ..., [n_harmonic]]]
    This function is LRU cached.
    Returns:
        multiplier: [1, n_harmonic, 1]
    """
    return torch.as_tensor(
        [[1.0 * k for k in range(1, n_harmonic + 1)]]
    ).reshape(1, n_harmonic, 1).to(device)


def freq_antialias_mask(sampling_rate: int, freq: Tensor,
                        hard_boundary: Optional[bool] = True) -> Tensor:
    """Return a harmonic amplitude mask that silence any harmonics above
    Nyquist frequency.
    Args:
        sampling_rate: The sampling rate in Hertz.
        freq: FloatTensor of any shape, values in Hertz.
    Returns:
        mask: Mask tensor of the same shape as freq_tensor.
            mask[freq_tensor > fs / 2] are zeroed.
    """
    if hard_boundary:
        return (freq < sampling_rate / 2.0).float()
    else:
        return torch.sigmoid(-(freq - sampling_rate / 2.0))


def harmonic_amplitudes_to_signal(f0_t: Tensor, harmonic_amplitudes_t: Tensor,
                                  sampling_rate: int, min_f0: float) -> Tensor:
    """Generate harmonic signal from given frequency and harmonic amplitudes.
    The phase of sinusoids are assumed to be all zero. The periodic function
    used is SINE.

    Args:
        f0_t: [n_batch, 1, n_sample]. Fundamental frequency per
            sampling point in Hertz.
        harmonic_amplitudes_t: [n_batch, n_harmonic, n_sample].
            Harmonic amplitudes per sampling point.
        sampling_rate: Sampling rate in Hertz.
        min_f0: Minimum f0 to accept. All f0_t below min_f0 are ignored.

    Returns:
        signal: [n_batch, 1, n_sample]. Sum of sinusoids with given harmonic
            amplitudes and fundamental frequencies.
    """
    _, n_harmonic, _ = harmonic_amplitudes_t.shape
    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
    # [n_batch, n_harmonic, n_sample]
    weight_map = (
        freq_antialias_mask(sampling_rate, f0_map) * harmonic_amplitudes_t
    )  # [n_batch, n_harmonic, n_sample]
    f0_map_cum = f0_t.cumsum(dim=-1) * freq_multiplier(
        n_harmonic, f0_t.device
    )  # [n_batch, n_harmonic, n_sample]
    w0_map_cum = f0_map_cum * 2.0 * pi / sampling_rate
    source = torch.sum(
        torch.sin(w0_map_cum) * weight_map, dim=-2, keepdim=True
    )  # [n_batch, 1, n_sample]
    source = (~(f0_t < min_f0)).float() * source
    return source * 0.01


def generate_impulse_train(f0_t: Tensor, n_harmonic: int,
                           sampling_rate: Union[int, float],
                           min_f0: Optional[float] = 1.0) -> Tensor:
    """Generate impulse train with sinusoidal synthesis.

    Args:
        f0_t: [n_batch, 1, n_sample]
        n_harmonic: Maximum number of harmonics in sinusoidal synthesis.
        sampling_rate: Sampling rate in Hertz.

    Returns:
        signal: [n_batch, 1, n_sample]
    """
    f0_map = freq_multiplier(n_harmonic, f0_t.device) * f0_t
    # [n_batch, n_harmonic, n_sample]
    weight_map = freq_antialias_mask(
        sampling_rate, f0_map
    )  # [n_batch, n_harmonic, n_sample]
    w0_map_cum = (
        f0_t.cumsum(dim=-1) * 2.0 * pi / sampling_rate *
        freq_multiplier(n_harmonic, f0_t.device)
    )  # [n_batch, n_harmonic, n_sample]
    source = torch.sum(torch.cos(w0_map_cum) * weight_map, dim=1, keepdim=True)
    source = (~(f0_t < min_f0)).float() * source
    return source * 0.01
