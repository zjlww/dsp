"""
Functions for interpolation in time.
"""
import torch
from torch import Tensor

from .utils import frame_signal, unframe_signal, get_window


def average_pool(x: Tensor, frame_size: int) -> Tensor:
    """Wrapper for 1D average pooling.
    Args:
        x: [n_batch, n_channel, n_sample]
    Returns:
        y: [n_batch, n_channel, n_sample // frame_size]
    """
    return torch.nn.functional.avg_pool1d(x, frame_size)


def linear_interpolate(x: Tensor, frame_size: int) -> Tensor:
    """Wrapper of 1D linear interpolation.
    Args:
        x: [n_batch, n_channel, n_frame]
    Returns:
        y: [n_batch, n_channel, n_frame * frame_size]
    """
    return torch.nn.functional.interpolate(x, scale_factor=frame_size,
                                           mode="linear", align_corners=False)


def zero_stuffing(x: Tensor, n_zeros: int) -> Tensor:
    """Stuff zeros between signal points in x.
    Args:
        x: [n_batch, 1, n_sample].
        n_zeros: How many zeros to insert between samples of x.
    Returns:
        y: [n_batch, 1, n_sample * (n_zeros + 1)].
    """
    n_batch, _, n_sample = x.shape
    zeros = torch.zeros(n_batch, n_zeros, n_sample, dtype=x.dtype,
                        device=x.device)
    framed_stuffed_x = torch.cat([x, zeros], dim=1)
    stuffed_x = unframe_signal(framed_stuffed_x, (n_zeros + 1))
    return stuffed_x


def window_interpolate(x: Tensor, window: Tensor,
                       frame_shift: int) -> Tensor:
    """Interpolate signal x by weight, overlap, and add window tensor.
    Similar to OLA ISTFT, the window tensor should satisfy COLA constraint
    under frame_shift. This function is equivalent to zero stuffing, then
    compute convolution of the window tensor with x.
    This function does not trim the excess at both end after OLA.
    Args:
        x: [..., n_frame].
        window: [window_size]
    Returns:
        y: [..., window_size + frame_shift * (n_frame - 1)]
    """
    batch_size = x.shape[:-1]
    n_frame = x.size(-1)
    squashed_x = x.reshape(-1, 1, n_frame)
    framed_interpolated_x = squashed_x * window.reshape(1, -1, 1)
    # shape: [prod(batch_size), window_size, n_frame]
    interpolated_x = unframe_signal(framed_interpolated_x, frame_shift)
    interpolated_x = interpolated_x.reshape(*batch_size, -1)
    return interpolated_x


def hann_window_interpolate(x: Tensor, frame_shift: int) -> Tensor:
    """A preset of the window interpolate function that uses hanning window.
    Excess on both ends are trimmed.
    Args:
        x: [..., n_frame]
    Returns:
        y: [..., n_frame * frame_shift]
    """
    window = get_window("hann", frame_shift * 2, x.device)
    y = window_interpolate(x, window, frame_shift)
    extra_length = y.size(-1) - x.size(-1) * frame_shift
    trimmed_y = y[..., extra_length // 2: -(extra_length - extra_length // 2)]
    return trimmed_y


def repeat_interpolate(x: Tensor, frame_size: int) -> Tensor:
    """Upsample by repeating in each frame.

    Args:
        x: [..., n_frame]
        frame_size: Frame size.
    Returns:
        y: [..., n_sample = n_frame * frame_size]
    """
    return torch.repeat_interleave(x, frame_size, -1)


def signal_median_filter(x: Tensor, filter_size: int) -> Tensor:
    """Median filter batched one dimensional signal.
    Args:
        x: [n_batch, 1, n_sample].
        filter_size: size of the median filter.
    Returns:
        y: [n_batch, 1, n_sample].
    """
    left_pad = filter_size // 2
    right_pad = filter_size // 2 - (filter_size + 1) % 2
    assert filter_size < x.size(-1), "Signal Not Long Enough"
    x = torch.nn.functional.pad(x, [left_pad, right_pad], "reflect")
    moving_window_x = frame_signal(x, filter_size, 1)
    y, _ = moving_window_x.median(dim=1, keepdim=True)
    return y


def median_filter_1d(x: Tensor, filter_size: int) -> Tensor:
    """One dimensional median filter with multi batch and channel input.
    Args:
        x: [n_batch, n_channel, n_frame].
        filter_size (int): size of the median filter.
    Returns:
        y: [n_batch, n_channel, n_frame].
    """
    n_batch, n_channel, n_frame = x.shape
    left_pad = filter_size // 2
    right_pad = filter_size // 2 - (filter_size + 1) % 2
    x = torch.nn.functional.pad(x, [left_pad, right_pad])
    x = x.reshape(n_batch, 1, -1)  # [n_batch, 1, n_channel x n_frame']
    x = signal_median_filter(x, filter_size)
    x = x.reshape(n_batch, n_channel, -1)
    x = x[:, :, left_pad:-right_pad]
    return x


def feature_moving_average(x: Tensor, weights: Tensor) -> Tensor:
    """Compute moving average of x with provided weights along the last
    dimension. Moving average is implemented with conv2d.
    Args:
        x: [n_batch, n_channel, n_frame].
        weight: [window_size].
    Returns:
        y: [n_batch, n_channel, n_frame].
    """
    window_size = len(weights)
    weights = weights / weights.sum()
    x = torch.nn.functional.pad(
        x, [window_size // 2, window_size - window_size // 2], mode="replicate"
    )
    x = torch.conv2d(x.unsqueeze(1), weights.view(1, 1, 1, -1))
    x = x.squeeze(1)
    return x


def hann_window_feature_moving_average(x: Tensor, window_size: int) -> Tensor:
    """Computing moving average of x with Hanning window along the last
    dimension.
    Args:
        x: [n_batch, n_channel, n_frame].
        wnidow_size: window_size of the hanning window.
    Returns:
        y: [n_batch, n_channel, n_frame].
    """
    weights = get_window("hann", window_size, x.device, False)
    return feature_moving_average(x, weights)
