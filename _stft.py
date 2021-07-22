"""
Depricated. Trying to reproduce PyTorch STFT behavior step by step.
"""
import torch
from torch import Tensor
from torch.nn.functional import pad

from typing import Optional

from .stfta import bare_stft

def stft_pytorch(
    x: Tensor,
    fft_size: int,
    window: Tensor,
    hop_size: Optional[int] = None,
    onesided: Optional[bool] = True,
    center: Optional[bool] = True,
    pad_mode: Optional[str] = "reflect",
    normalized: Optional[bool] = False,
    use_pytorch_implementation: Optional[bool] = False,
) -> Tensor:
    """STFT implementation that is a subset of the PyTorch implementation.
    This function is a few times slower than then monolithic implementation
    in native C. The window is always padded on both side, to be at the
    center of (fft_size) frame. The signal is padded on both side when
    center is True.

    This function handles window padding automatically.

    Args:
        x: [..., n_sample].
        fft_size: NFFT.
        window: Window tensor of shape (window_size).
        hop_size: Hop size. Defaults to fft_size // 4.
        center: Pad input fft_size // 2 on both side of x. Defaults to True.
        pad_mode: How to pad x when center is true. Defaults to 'reflect'.
        normalized: Whether the result is scaled by (fft_size)**-0.5.
                    Defaults to False.
        onesided: Whether the FFT values are onesided. Defaults to True.
        use_pytorch_implementation: Whether to use the PyTorch implementation.
                                    Defaults to False.

    Returns:
        X: [..., n_frame, fft_size, 2] or
           [..., n_frame, fft_size // 2 + 1, 2] when onesided is True.
    """
    if hop_size is None:
        assert fft_size >= 4, "fft_size too small"
        hop_size = fft_size // 4
    window_size = len(window)
    assert window_size <= fft_size, "window_size > fft_size"

    batch_size = x.shape[:-1]
    n_sample = x.size(-1)
    squashed_x = x.reshape(-1, n_sample)

    if use_pytorch_implementation:
        squashed_X = torch.stft(
            squashed_x,
            fft_size,
            hop_size=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
        ).transpose(-2, -3)
    else:
        lpad = fft_size // 2
        rpad = fft_size // 2
        wlpad = (fft_size - window_size) // 2
        wrpad = fft_size - window_size - wlpad
        if center:
            squashed_x = pad(squashed_x.unsqueeze(-2),
                             [lpad, rpad], mode=pad_mode).squeeze(-2)
        if normalized:
            squashed_x = squashed_x * (fft_size ** -0.5)
        padded_window = pad(window, [wlpad, wrpad], mode="constant")
        squashed_X = bare_stft(squashed_x, padded_window, hop_size, onesided)
    X = squashed_X.reshape(*batch_size, -1, *squashed_X.shape[-2:])
    return X