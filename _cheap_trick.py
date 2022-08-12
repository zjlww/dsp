"""
A differentiable implementation of CheapTrick
Matlab Version by M. Morise
"""
# pylint: disable=E1101

import torch
import torch.nn as nn
from math import ceil, floor, log2
from torch.nn.functional import pad
from modules.dynamic_filter import frame_signal, unframe_signal, channel_conv
import numpy as np


class CheapTrick:
    """
    Gradient only passed to waveform, f0 will not receive gradient.
    Reference:
        Digital Frequency = f = w / 2Pi
        Analog Frequency = F = W / 2Pi
        Sampling Period = T = 1 / Fs
        Sampling Frequency = Fs = 16000
        Connections:
            w = WT = W / Fs = 2 Pi F / Fs
            f = FT = F / Fs 
    """

    def __init__(
        self,
        hop_length,
        min_f0=71,
        max_f0=800,
        default_f0=500,
        fs=16000,
        q1=-0.15,
        fft_size=None
        ):
        self.fs = fs
        self.hop_length = hop_length
        self.default_f0 = default_f0
        self.fft_size = 2 ** ceil(log2(3 * fs / min_f0 + 1))
        if fft_size is not None:
            assert fft_size >= self.fft_size, "Too Small FFT Size"
            assert fft_size % 2 == 0, "FFT Size must be even number"
            self.fft_size = fft_size
        self.q1 = -0.15
        self.min_f0 = fs * 3.0 / (self.fft_size - 3.0)
        self.max_f0 = max_f0
        self.max_period = ceil(fs / float(self.min_f0))
        # The Max Window Length is guaranteed to be odd
        self.max_window_length = 3 * self.max_period // 2 * 2 + 1
        self.half_max_window_length = self.max_window_length // 2
        assert self.max_window_length <= self.fft_size, "FFT Size < Window Length"
        self.max_smooth_kernel_length = ceil(self.fft_size * self.max_f0 * 0.667 / self.fs) // 2 * 2 + 1
        self.half_smooth_kernel_length = self.max_smooth_kernel_length // 2

    def frame_waveform(self, x):
        """
        Either end are zero padded......
        x: [Batch, 1, T = Hop x N]
        returns: [Batch, MaxW, N = T // Hop]
        """
        x = pad(x, [self.half_max_window_length, self.half_max_window_length], mode="constant")
        # [Batch, 1, Hop x N + 2 HalfM]
        framed_x = frame_signal(x, self.max_window_length, self.hop_length)
        return framed_x
    
    def hamming_window(self, x):
        """
        x: [...]
        returns: [...]
        """
        with torch.no_grad():
            y = x
            z = torch.cos(x * np.pi) * 0.5 + 0.5
            z[y > 1.0] = 0.0
            z[y < -1.0] = 0.0
            return z
    
    def sinc(self, x):
        """
        x: [...]
        returns: [...]
        It returns NAN at x = 0
        """
        with torch.no_grad():
            return torch.sin(x) / x
    
    def get_frame_centered_windows(self, f0):
        """
        f0: [Batch, 1, N]
        returns: [Batch, MaxW, N]
        """
        batch, _, n = f0.shape
        with torch.no_grad():
            meter = torch.linspace(
                -self.half_max_window_length,
                self.half_max_window_length,
                self.max_window_length,
                device=f0.device
            ).reshape(1, self.max_window_length, 1).repeat(batch, 1, n) # [Batch, MaxW, N]
        
            half_window_size = self.fs / f0 * 1.5 # [Batch, 1, N]
            meter = meter / half_window_size
            window = self.hamming_window(meter) # [Batch, MaxW, N]
            window = window / (window.pow(2).sum(dim=1, keepdim=True).sqrt())
        return window
    
    def window_framed_waveform(self, framed_x, f0):
        """
        x: [Batch, MaxW, N]
        f0: [Batch, 1, N]
        Note: frame center of x is at [..., HalfW, ...]
        Note: f0 of unvoiced part are fixed to default value
        returns: [Batch, MaxW, N]
        """
        window = self.get_frame_centered_windows(f0)
        windowed_signal = window * framed_x
        return windowed_signal
    
    def get_power_spectrum(self, windowed_x):
        """
        windowed_x: [Batch, MaxW, N]
        returns: [Batch, N, NFFT]
        """
        windowed_x = windowed_x.transpose(-1, -2) 
        windowed_x = pad(windowed_x, [0, self.fft_size - self.max_window_length])
        X = torch.rfft(windowed_x, signal_ndim=1, normalized=False, onesided=False)
        # [Batch, N, NFFT, 2]
        P = X[..., 0] ** 2 + X[..., 1] ** 2 # [Batch, N, NFFT]
        return P

    def get_frame_centered_smoothing_kernels(self, f0):
        """
        f0: [Batch, 1, N] 
        returns: [Batch, MaxS, N]
        """
        batch, _, n = f0.shape
        with torch.no_grad():
            meter = torch.linspace(
                -self.half_smooth_kernel_length * self.fs / self.fft_size,
                self.half_smooth_kernel_length * self.fs / self.fft_size,
                self.max_smooth_kernel_length,
                device=f0.device
            ).reshape(1, self.max_smooth_kernel_length, 1).repeat(batch, 1, n) # [Batch, MaxW, N]
            k = self.fft_size / self.fs
            b = (f0 / 3 + 1) * k # [Batch, 1, N]
            y = torch.clamp(- meter.abs() * k + b, 0, 1) # [Batch, MaxS, N]
        return y
    
    def smooth_power_spectrum(self, P, f0):
        """
        Uses Channel Convolution to linear smooth power spectrum
        The implementation is not matching WORLD for now.
        Smooth from - 1/3 f0 ~ 1/3 f0
        P: [Batch, N, NFFT]
        f0: [Batch, 1, N]
        returns: [Batch, N, NFFT]
        """
        smooth_kernel = self.get_frame_centered_smoothing_kernels(f0)
        Ps = channel_conv(P.transpose(-1, -2), smooth_kernel.transpose(-1, -2), pad_mode="circular").transpose(-1, -2)
        Ps = Ps[..., self.half_smooth_kernel_length: -self.half_smooth_kernel_length]
        Ps = Ps + 1e-16
        return Ps

    def liftering(self, Ps, f0):
        """
        Liftering in Cepstral Domain
        Ps: [Batch, N, NFFT]
        f0: [Batch, 1, N]
        returns: [Batch, N, NFFT // 2 + 1]
        """
        with torch.no_grad():
            f0 = f0.squeeze(1).unsqueeze(-1) # [Batch, N, 1]
            quefrency_axis = torch.linspace(0, self.fft_size - 1, self.fft_size, device=Ps.device) / self.fs # [NFFT]
            quefrency_axis[self.fft_size // 2 + 1:] = quefrency_axis[1:self.fft_size // 2].flip(0)
            # Smoothing Filter
            smoothing_lifter = self.sinc(np.pi * quefrency_axis * f0).unsqueeze(-1) # [Batch, N, NFFT, 1]
            smoothing_lifter[:, :, 0, :] = 1.0
            compensation_lifter = (1 - 2 * self.q1) + 2 * self.q1 * torch.cos(2 * np.pi * quefrency_axis * f0).unsqueeze(-1)
        
        cepstrum = torch.rfft(torch.log(Ps), signal_ndim=1, onesided=False) # [Batch, N, NFFT, 2]
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter
        spectral_envelope = torch.exp(torch.irfft(liftered_cepstrum, signal_ndim=1, onesided=False)) # [Batch, N, NFFT]
        return spectral_envelope

    def spectral_envelope(self, x, f0):
        """
        x: [Batch, 1, T = N x Hop]
        f0: [Batch, 1, N = T // Hop]
        returns: [Batch, N, FFT // 2 + 1] in power spectrum
        """
        f0 = f0.clone()
        f0[f0 < self.min_f0] = self.default_f0
        framed_x = self.frame_waveform(x)
        windowed_x = self.window_framed_waveform(framed_x, f0)
        P = self.get_power_spectrum(windowed_x)
        Ps = self.smooth_power_spectrum(P, f0)
        Pe = self.liftering(Ps, f0)
        return P[..., :self.fft_size // 2 + 1], \
            Ps[..., :self.fft_size // 2 + 1], \
            Pe[..., :self.fft_size // 2 + 1]
    


if __name__ == "__main__":
    from dataset import LJSLoader
    loader = LJSLoader(16000, 80, 71, 800, "/home/sorcerer/Datasets/jsut_ver1.1")
    wav, f0 = loader.sample()
    wav = wav.reshape(1, 1, -1)
    f0 = f0.reshape(1, 1, -1)
    trick = CheapTrick(80, 71, 800, 500, 16000)
    P = trick.spectral_envelope(wav, f0)
