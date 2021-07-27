"""
Filter banks.
"""
import torch
from torch import Tensor
from torch.nn.functional import pad

from .utils import frame_signal

from typing import Tuple, Callable, Iterable
from math import ceil, pi


@torch.no_grad()
def continuous_hann_window(x: Tensor) -> Tensor:
    y = x
    z = torch.cos(x * pi) * 0.5 + 0.5
    z[y > 1.0] = 0.0
    z[y < -1.0] = 0.0
    return z


def window_based_filter_bank(
    center_freqs: Tensor,
    window_lengths: Iterable[int],
    window_function: Callable[[Tensor], Tensor],
    sampling_rate: int
) -> Tuple[Tensor, Tensor]:
    """Creates the linear transform matrix of a filter bank. All filters are
    created by modulating a common window function to shift their center
    frequencies. The length of windows can be changed.
    Args:
        center_freqs: The center frequencies of the filters.
        window_lengths: The window function in [-1, 1] will be stretched to
                        [-wl, wl]
        window_function: The windowing function, [-1, 1] -> R,
        sampling_rate: int
    Returns:
        basis: Complex Tensor [max_window_length, n_filter].
    """
    window_lengths = list(window_lengths)
    n_filter = len(window_lengths)
    max_window_length = int(ceil(max(window_lengths)) // 2 * 2 + 1)
    window_lengths = torch.as_tensor(
        window_lengths, dtype=torch.float32, device=center_freqs.device)
    half_max_window_length = int(max_window_length // 2)
    grid = torch.linspace(
        -half_max_window_length,
        half_max_window_length,
        max_window_length,
        device=center_freqs.device
    ).reshape(1, -1).repeat(n_filter, 1)  # [n_filter, max_width]
    meter = grid / window_lengths.unsqueeze(-1)  # [n_filter, max_width]
    windows = window_function(meter) / window_lengths.unsqueeze(1)
    # [n_filter, max_width] centered windows, weight corrected by window length
    freq_grid = grid * (-2 * pi * center_freqs.unsqueeze(-1) /
                        sampling_rate)
    # [n_filter, max_width]
    real_basis = torch.cos(freq_grid) * windows  # [n_filter, max_width]
    imag_basis = torch.sin(freq_grid) * windows  # [n_filter, max_width]
    return torch.complex(real_basis.T, imag_basis.T)


def apply_filter_bank(x: Tensor, hop_size: int, basis: Tensor):
    """Apply a filter bank on a signal.
    Args:
        x: Complex Tensor [..., n_sample]
        basis: Complex Tensor [max_window_length, n_filter]
    Returns:
        X: Complex Tensor [..., n_frame, n_filter]
    """
    batch_size = x.shape[:-1]
    n_sample = x.size(-1)
    max_window_length = basis.size(0)
    padded_x = pad(
        x.reshape(-1, 1, n_sample), [
            max_window_length // 2,
            max_window_length - max_window_length // 2 - 1
        ]
    )
    framed_x = frame_signal(padded_x, max_window_length, hop_size)
    framed_x = framed_x.transpose(-1, -2)
    X = torch.matmul(framed_x, basis)
    # [prod(*batch_size), n_frame, n_filter]
    return X.reshape(*batch_size, *X.shape[1:])
