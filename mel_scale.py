"""
Mel-scale definition.
"""
import torch
from torch import Tensor
from typing import Union
import numpy as np
from math import log
import librosa
from librosa.filters import mel as mel_fn


def hz_to_mel(
    frequencies: Union[float, int, Tensor, np.ndarray],
        htk=False) -> Union[float, int, Tensor, np.ndarray]:
    """Convert Hz to Mels.
    Extending librosa.hz_to_mel to accepting Tensor.
    """
    if not isinstance(frequencies, Tensor):
        return librosa.hz_to_mel(frequencies)
    if htk:
        return 2595.0 * torch.log10(1.0 + frequencies / 700.0)

    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = log(6.4) / 27.0  # step size for log region
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + torch.log(frequencies[log_t] / min_log_hz) / \
        logstep
    return mels


def mel_to_hz(
    mels: Union[int, float, Tensor, np.ndarray],
        htk=False) -> Union[int, float, Tensor, np.ndarray]:
    """Convert mel bin numbers to frequencies."""
    if not isinstance(mels, Tensor):
        return librosa.mel_to_hz(mels, htk=htk)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = log(6.4) / 27.0  # step size for log region
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * \
        torch.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def linear_mel_matrix(
    sampling_rate: int, fft_size: int, mel_size: int,
    mel_min_f0: Union[int, float],
    mel_max_f0: Union[int, float],
    device: torch.device
) -> Tensor:
    """
    Args:
        sampling_rate: Sampling rate in Hertz.
        fft_size: FFT size, must be an even number.
        mel_size: Number of mel-filter banks.
        mel_min_f0: Lowest frequency in the mel spectrogram.
        mel_max_f0: Highest frequency in the mel spectrogram.
        device: Target device of the transformation matrix.

    Returns:
        basis: [mel_size, fft_size // 2 + 1].
    """
    basis = torch.FloatTensor(
        mel_fn(sampling_rate, fft_size, mel_size, mel_min_f0, mel_max_f0)
    ).transpose(-1, -2)
    return basis.to(device)