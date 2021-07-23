import torch
from torch import Tensor
from torch.nn.functional import pad

from math import log

from typing import Optional, Union, Tuple


def mulaw(x: Tensor, mu: Union[float, int]) -> Tensor:
    return torch.sign(x) * torch.log(1 + mu * x.abs()) / log(1 + mu)


def reshape_zeros_like(x: Tensor, dim: int, length: int) -> Tensor:
    """Return torch.zeros_like(x), while change shape of the `dim` dimension."""
    shape = list(x.shape)
    shape[dim] = length
    return torch.zeros(*shape, dtype=x.dtype, device=x.device)


def fftpad(x: Tensor, padding: int, dim: Optional[int] = -1) -> Tensor:
    """Insert zeros in x with the following pattern:
        [x, y] => [x, 0, y];
        [x, y, z] => [x, y, 0, z];
    """
    size = x.size(dim)
    half = size // 2
    first_half = torch.narrow(x, dim, 0, size - half)
    second_half = torch.narrow(x, dim, size - half, half)
    zeros = reshape_zeros_like(x, dim, padding)
    return torch.cat([first_half, zeros, second_half], dim=dim)


def ifftpad(x: Tensor, size: int, dim: Optional[int] = -1) -> Tensor:
    """Inverse of fftpad function."""
    half = size // 2
    first_half = torch.narrow(x, dim, 0, size - half)
    second_half = torch.narrow(x, dim, x.size(dim) - half, half)
    return torch.cat([first_half, second_half], dim=dim)


def get_window(window_type: str,
               window_size: int,
               device: Union[str, torch.device],
               periodic: Optional[bool] = True,
               padding: Optional[Tuple[int, int]] = (0, 0)
               ) -> Tensor:
    """LRU cached window functions. Wrapper around PyTorch functions.
    Examples:
        >>> get_window("hann", 10, "cpu", periodic=True)
        tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, \
            0.6545, 0.3455, 0.0955])
    """
    window = getattr(torch, f"{window_type}_window")(
        window_size, device=device, periodic=periodic
    )
    window = pad(window, padding, mode="constant", value=0.0)
    return window


def cyclic_narrow(x: Tensor, dim: int, start: int, length: int) -> Tensor:
    """Cyclic version of torch.narrow()."""
    end = start + length
    size = x.size(dim)
    if end < size:
        return torch.narrow(x, dim, start, length)
    else:
        a = torch.narrow(x, dim, start, size - start)
        b = torch.narrow(x, dim, 0, end - size)
        return torch.cat([a, b], dim=dim)


def unframe_signal(x: Tensor, frame_shift: int) -> Tensor:
    """This function uses overlap and add to unframe a framed signal.
    Shapes:
        x: [n_batch, frame_size, n_frame]
        returns: [n_batch, 1, frame_size + (n_frame - 1) * frame_shift]
    """
    _, frame_size, n_frame = x.shape
    n_sample = frame_size + (n_frame - 1) * frame_shift
    return torch.nn.functional.fold(
        x,
        output_size=(n_sample, 1),
        kernel_size=(frame_size, 1),
        stride=(frame_shift, 1),
    ).squeeze(-1)


def frame_signal(x: Tensor, frame_size: int, frame_shift: int) -> Tensor:
    """Frame a signal with given frame size and frame shift.
    The first frame starts from 0. When the signal x is not long enough
    to fill a frame, the sampling points are dropped.
    You should pad appropriately to preserve these sampling points.
    NOTE: n_sample >= frame_size must be true

    Args:
        x: [n_batch, 1, n_sample]

    Returns:
        framed_x (Tensor):
        [n_batch, frame_size, (n_sample - frame_size + frame_shift) // frame_shift]
    """
    return torch.nn.functional.unfold(
        x.unsqueeze(-1), kernel_size=(frame_size, 1), stride=(frame_shift, 1)
    )
