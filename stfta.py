"""
STFT Analysis.
"""
from typing import Iterable
import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.fft import fft

from .utils import get_window, frame_signal
from .mel_scale import linear_mel_matrix
from .bank import window_based_filter_bank, apply_filter_bank, continuous_hann_window


def bare_stft(x: Tensor, padded_window: Tensor, hop_size: int) -> Tensor:
    """Compute STFT of real 1D signal.
    This function does not handle padding of x, and the window tensor.
    This function assumes fft_size = window_size.
    Args:
        x: [..., n_sample]
        padded_window: [fft_size], a window padded to fft_size.
        hop_size: Also referred to as the frame shift.
    Returns:
        n_frame: see frame_signal definition.
        X: [..., n_frame, fft_size],
            where n_frame = n_sample // hop_size
    """
    fft_size = len(padded_window)
    # Squash x's batch_sizes
    batch_size = x.shape[:-1]
    n_sample = x.size(-1)
    squashed_x = x.reshape(-1, 1, n_sample)
    # shape: [prod(batch_size), 1, n_sample]
    framed_squashed_x = frame_signal(squashed_x, fft_size, hop_size)
    # shape: [prod(batch_size), fft_size, n_frame]
    windowed_framed_squashed_x = \
        framed_squashed_x * padded_window.unsqueeze(-1)
    squashed_X = fft(
        windowed_framed_squashed_x.transpose(-1, -2), dim=-1
    )  # shape: [prod(batch_size), n_frame, fft_size]
    X = squashed_X.reshape(*batch_size, *(squashed_X.shape[1:]))
    # shape: [*batch_size, n_frame, fft_size]
    return X


def frame_center_stft(
    x: Tensor,
    hop_size: int,
    window_size: int,
    window_type: str,
    fft_size: int,
) -> Tensor:
    """STFT analysis where the window is located at around the center
    of each frame.
    Args:
        x: [n_batch, 1, n_sample].
        window_type: STFT window type. See `get_window` for details.
    Returns:
        X: [n_batch, n_frame, fft_size], 
            where n_frame = n_sample // hop_size
    """
    assert fft_size >= window_size >= hop_size

    padding_left = window_size // 2 - hop_size // 2
    padding_right = fft_size - padding_left - hop_size
    window = get_window(
        window_type, window_size, x.device, periodic=False,
        padding=(0, fft_size - window_size),
    )
    x = pad(x, [padding_left, padding_right])
    X = bare_stft(x, window, hop_size)
    return X.squeeze(1)


def stft_magnitude_to_mel_scale_log_magnitude(
    sampling_rate: int,
    mag_X: Tensor,
    fft_size: int,
    mel_size: int,
    mel_min_f0: float,
    mel_max_f0: float,
    mel_log_min_clip: float
) -> Tensor:
    """Project onesided STFT magnitude to mel scale log magnitude.
    Args:
        sampling_rate: Sampling rate in Hertz.
        mag_X: [..., fft_size].
        fft_size: The fft_size of mag_X.
        log_mel_min_clip: A (mostly negative) float number to clip
                          the log mel scaled spectrogram.
    Returns:
        mel_magnitude_x: [..., mel_size].
    """
    assert fft_size % 2 == 0
    projection_matrix = linear_mel_matrix(
        sampling_rate, fft_size, mel_size, mel_min_f0,
        mel_max_f0, mag_X.device)
    return mag_X[..., :(fft_size // 2 + 1)].matmul(projection_matrix) \
        .log().clamp(min=mel_log_min_clip)


def frame_center_log_mel_spectrogram(
    x: Tensor, hop_size: int, window_size: int,
    window_type: str, fft_size: int, sampling_rate: int,
    mel_size: int, mel_min_f0: float, mel_max_f0: float,
    mel_log_min_clip: float
) -> Tensor:
    """Frame center log mel spectrogram.
    Args:
        x: [n_batch, 1, n_sample].
    Returns:
        mag_X: [n_batch, n_frame, mel_size]
    """
    X = frame_center_stft(
        x, hop_size, window_size, window_type, fft_size
    )
    mag_X = X.abs()
    return stft_magnitude_to_mel_scale_log_magnitude(
        sampling_rate, mag_X, fft_size, mel_size, mel_min_f0, mel_max_f0,
        mel_log_min_clip)


def cfbstft(
    x: Tensor, window_size: float, hop_size: int,
    n_filter: int, min_f0: float, max_f0: float,
    sampling_rate: int,
):
    """Continuous filter bank based STFT. 
    Args:
        x: [..., n_sample].
        hop_size: time sampling in time-frequency analysis.
        n_filter: int, number of filters.
    Returns:
        X: Complex Tensor [..., n_frame, n_filter]
    """
    center_freqs = torch.linspace(min_f0, max_f0, n_filter, device=x.device)
    window_lengths = torch.ones_like(center_freqs) * window_size
    basis = window_based_filter_bank(
        center_freqs, window_lengths, continuous_hann_window, sampling_rate
    )
    return apply_filter_bank(x, hop_size, basis)


def stft_loss(
    x: Tensor, y: Tensor, fft_lengths: Iterable[int], window_lengths: Iterable[int], hop_lengths: Iterable[int], 
    loss_scale_type: str) -> Tensor:
    """Compute STFTLoss. The length of provided configuration lists should be
    the same.
    Args:
        x, y: [n_batch, 1, n_sample]
    Returns:
        loss: []
    """
    x, y = x.squeeze(1), y.squeeze(1)
    loss = 0.0
    batch_size = x.size(0)
    z = torch.cat([x, y], dim=0)  # shape: [2 x Batch, T]
    for fft_length, window_length, hop_length in zip(fft_lengths, window_lengths, hop_lengths):
        window = torch.hann_window(window_length, device=x.device)
        Z = torch.stft(z, fft_length, hop_length, window_length, window, return_complex=False)
        # shape: [2 x Batch, Frame, 2]
        SquareZ = Z.pow(2).sum(dim=-1) + 1e-10  # shape: [2 x Batch, Frame]
        SquareX, SquareY = SquareZ.split(batch_size, dim=0)
        MagZ = SquareZ.sqrt()
        MagX, MagY = MagZ.split(batch_size, dim=0)
        if loss_scale_type == "log_linear":
            loss += (MagX - MagY).abs().mean() + \
                    0.5 * (SquareX.log() - SquareY.log()).abs().mean()
        elif loss_scale_type == "linear":
            loss += (MagX - MagY).abs().mean()
        elif isinstance(loss_scale_type, float):
            loss += (MagX - MagY).abs().mean() + \
                    0.5 * loss_scale_type * \
                          (SquareX.log() - SquareY.log()).abs().mean()
        else:
            raise RuntimeError(f"Unrecognized STFT loss scale type.")
    return loss
