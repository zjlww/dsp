from math import log2, ceil
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import pad
from torch.fft import fft, fftshift, ifft

from .utils import unframe_signal, frame_signal, get_window


def time_corr(x: Tensor, y: Tensor,
              flip: Optional[bool] = False) -> Tensor:
    """Compute batched 1D correlation of signal x and signal y in the
    time domain.

    Shapes:
        shape of x and y must be broadcastable except for the last dimension.
        x: [..., nx]
        y: [..., ny]
        returns: [..., nx + ny - 1]

    Args:
        flip (bool, optional): Defaults to False. When set to True,
        framed_y is flipped in time.
    """
    # Implemented with dark magic. Must broadcast in the first step.
    x = x.view(*x.shape, 1)
    y = y.view(*y.shape[:-1], 1, y.size(-1))
    x, y = torch.broadcast_tensors(x, y)
    x = x[..., 0]
    y = y[..., 0, :]
    # End reshaping.
    nx = x.size(-1)
    batch_size = x.shape[:-1]
    multiplied_batch_size = 1
    for sz in batch_size:
        multiplied_batch_size *= sz
    ny = y.size(-1)
    if flip:
        y = y.flip(-1)
    x = x.reshape(1, multiplied_batch_size, nx)  # (1, *batch_size, nx)
    x = pad(x, [ny - 1, ny - 1], mode="constant")
    y = y.reshape(multiplied_batch_size, 1, ny)  # (*batch_size, 1, ny)
    framed_z = torch.nn.functional.conv1d(x, y, groups=multiplied_batch_size)
    framed_z = framed_z.reshape(*batch_size, -1)
    return framed_z


def fft_multiply(x: Tensor, y: Tensor,
                 flip: Optional[bool] = False) -> Tensor:
    """Computes convolution in the frequency domain with FFT multiplication.
    As the function operates in the frequency domain, the input shape
    is treated as the fft_size. Since FFT is used, the result is
    time aliased. Input should be padded appropriately to avoid aliasing.

    Shapes:
        x: [..., fft_size]
        y: [..., fft_size]
        returns: [..., fft_size]

    Args:
        flip: If flip is set to True, the sign of the
        imaginary part of the second signal's FFT is flipped. When flipped,
        this function actually computes correlation.
    """
    fft_size = x.size(-1)
    assert fft_size == y.size(-1), \
        "Size of the last dimension in two tensor does not match."
    fft_x = fft(x, dim=-1)
    fft_y = fft(y, dim=-1)
    flip_sign = -1 if flip else +1
    fft_z_real = fft_x.real() * fft_y.real() - \
        flip_sign * fft_x.imag() * fft_y.imag()
    fft_z_imag = flip_sign * fft_x.real() * fft_y.imag() + \
        fft_x.imag() * fft_y.real()
    fft_z = torch.complex(fft_z_real, fft_z_imag)
    z = ifft(fft_z, dim=-1).real()
    return z


def fft_corr(x: Tensor, y: Tensor, flip: Optional[bool] = False,
             fft_size: Optional[int] = None) -> Tensor:
    """Compute batched 1D correlation of signal x and y in the frequency
    domain. Compared to fft_multiply, this function adds appropriate
    padding to avoid aliasing. This function has the same function as
    time_corr.

    Shapes:
        x: [..., nx]
        y: [..., ny]
        returns: [..., nx + ny - 1]

    Args:
        fft_size (int, optional): Defaults to None.
            When set, it override default fft_size.
        flip (bool, optional): Defaults to False. When set to True,
        this function computes convolution.
    """
    nx = x.size(-1)
    ny = y.size(-1)
    if fft_size is None:
        fft_size = 2 ** ceil(log2(nx + ny - 1))
    else:
        assert fft_size >= (nx + ny - 1)
    padded_y = pad(y, [0, fft_size - ny])
    if flip:
        padded_x = pad(x, [0, fft_size - nx])
    else:
        padded_x = pad(x, [ny - 1, fft_size - nx - ny + 1])
    z = fft_multiply(padded_x, padded_y, flip=not flip)
    return z[..., :nx + ny - 1]


def _framewise_corr_ola(
    framed_x: Tensor, framed_y: Tensor, frame_shift: int,
    method: Optional[str] = "time", flip: Optional[bool] = False) -> Tensor:
    """Computes time-varying 1D correlation in time or frequency domain.
    This function receives framed signal x. It filters framed signal x,
    and unframe the signal with OLA. This function preserves all
    non-zero entries, which is commonly referred to as the 'valid'
    padding.

    Args:
        framed_x: [n_batch, n_frame, nx]
        framed_y: [n_batch, n_frame, ny]
        method (str, optional): Either 'time' or 'fft'.
        flip (bool, optional): Defaults to False. When set to True, the
        signal y is flipped in time in each frame.

    Returns:
        z: [n_batch, 1, (nx + ny - 1) + (n_frame - 1) * frame_shift]
    """
    if method == "time":
        framed_z = time_corr(framed_x, framed_y, flip=flip)
    elif method == "fft":
        framed_z = fft_corr(framed_x, framed_y, flip=flip)
    else:
        raise ValueError(f"Convolution method {method} not implemented.")
    return unframe_signal(framed_z.transpose(-1, -2), frame_shift)


def ltv_fir(
    x: Tensor, filters: Tensor, frame_size: int, method: Optional[str] = "fft"
) -> Tensor:
    """Linear time-varying FIR filter with a square OLA window.
    Notice that this implements a convolution rather than a correlation.

    Args:
        x: [n_batch, 1, n_sample]
        filters: [n_batch, n_frame, filter_size].
                 Filter FIRs. FIRs in each frame are assumed to be stored
                 as time wrapped signals. Function fftshift convert it
                 back to continuous time order.
        frame_size: The frame size in sampling points.
        method: Either 'time' or 'fft'. Defaults to 'fft'.

    Returns: [n_batch, 1, n_sample]
             n_sample: n_frame * frame_size
    """
    filter_size = filters.size(-1)
    n_sample = x.size(-1)
    framed_x = frame_signal(x, frame_size, frame_size).transpose(-1, -2)
    # [n_batch, n_frame, frame_size]
    filters = fftshift(filters, dim=-1)
    y = _framewise_corr_ola(framed_x, filters, frame_size, method, flip=True)
    striped_y = y[..., filter_size // 2: n_sample + filter_size // 2]
    return striped_y


def windowed_ltv_fir(
    x: Tensor,
    window: Tensor,
    filters: Tensor,
    frame_size: int,
    method: Optional[str] = "fft"
) -> Tensor:
    """Linear time-varying FIR filter with arbirartary window. The window is
    a tensor of shape [window_size].
    FIXME: Boundary of both end are not correct due to windowing. Add extra
    padding to fix this.
    """
    filter_size = filters.size(-1)
    window_size = len(window)
    n_sample = x.size(-1)
    wlpad = window_size // 2
    wrpad = window_size - wlpad - 1
    padded_x = pad(x, [wlpad, wrpad], mode="constant", value=0.0)
    # shape: [n_batch, 1, n_sample + window_size - 1]
    framed_padded_x = frame_signal(padded_x, window_size,frame_size).transpose(-1, -2)
    windowed_framed_padded_x = framed_padded_x * window
    # shape: [n_batch, window_size, n_frame]
    filters = fftshift(filters, dim=-1)
    y = _framewise_corr_ola(windowed_framed_padded_x, filters,
                       frame_size, method, flip=True)
    striped_y = y[..., filter_size // 2 + wlpad: n_sample +
                  filter_size // 2 + wlpad]
    return striped_y


def hann_ltv_fir(
    x: Tensor,
    filters: Tensor,
    frame_size: int,
    method: Optional[str] = "fft"
) -> Tensor:
    """Windowed LTV-FIR with Hanning windows of twice the frame_size."""
    window = get_window("hann", frame_size * 2, x.device)
    return windowed_ltv_fir(x, window, filters, frame_size, method)


def fir(x: Tensor, filters: Tensor, method: Optional[str] = "fft",
        is_causal: Optional[bool] = False) -> Tensor:
    """Finite impulse response time invariant filter. Wrapper around
    time_corr and fft_corr functions.

    Args:
        x: [..., n_sample]
        filters: [..., filter_size]
            When is_causal is set to True, it uses normal time indexing.
            When is_causal is set to False, it uses wrapped time indexing.
        method (str, optional): fft or time. Defaults to "fft".
        is_causal: Defaults to False.

    Returns:
        y: [*x.shape]
    """
    n_sample = x.size(-1)
    filter_size = filters.size(-1)
    if not is_causal:
        filters = fftshift(filters, dim=-1)
    if method == "time":
        y = time_corr(x, filters, flip=True)
    elif method == "fft":
        y = fft_corr(x, filters, flip=True)
    else:
        raise ValueError(f"Method {method} not implemented.")
    if is_causal:
        return y[..., :n_sample]
    else:
        return y[..., filter_size // 2: n_sample + filter_size // 2]
