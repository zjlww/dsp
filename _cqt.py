import torch
from torch import Tensor
from math import pi, log2

from .bank import apply_filter_bank, window_based_filter_bank, continuous_hann_window


def cqt_basis(
	n_filter: int, Q: float, min_f0: float, max_f0: float,
	sampling_rate: int, device: torch.device
) -> Tensor:
    """Returns the real and imaginary basis of CQT.
    Args:
        n_filter: number of filter banks.
        Q: Q value of CQT.
    Returns: all tensors on device.
        basis: Complex Tensor [max_window_length, n_filter]
        window_lengths: [n_filter]
        filter_center_freqs: [n_filter]
    """
    # Compute the center frequency of the filters.
    log_min_f0 = log2(min_f0)
    log_max_f0 = log2(max_f0)
    filter_center_log_freq = torch.linspace(log_min_f0, log_max_f0, n_filter)
    filter_center_freqs = 2 ** filter_center_log_freq
    window_lengths = sampling_rate * Q / filter_center_freqs
    # The max window length is guaranteed to be odd
    basis = window_based_filter_bank(
        filter_center_freqs,
        window_lengths,
        continuous_hann_window,
        sampling_rate
    )
    return basis.to(device), window_lengths.to(device), filter_center_freqs.to(device)


def cqt(
    x: Tensor, hop_size: int, n_filter: int,
    Q: float, min_f0: float, max_f0: float,
    sampling_rate: int,
) -> Tensor:
    """CQT transformation. This function is implemented with linear projection.
    Args:
        x: [..., n_sample].
        n_filter: int,
        Q: float,
        min_f0: float,
        max_f0: float,
        sampling_rate: int
        max_window_length: restriction on the window length
    Returns:
        cqt_X: Complex Tensor [..., n_frame, n_filter]
    """
    basis, _, _ = cqt_basis(n_filter, Q, min_f0, max_f0, sampling_rate, x.device)
    return apply_filter_bank(x, hop_size, basis)
