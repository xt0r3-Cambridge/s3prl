# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ spectrogram.py ]
#   Synopsis     [ Spectrogram Converter implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Spectrogram implementation for SUPERB Author: Kornel Szabo, University of Cambridge

###############
#   IMPORTS   #
###############
# -------------#
import torch
import einops

# -------------#
from torch import Tensor
from typing import List
from jaxtyping import Float, Int64, jaxtyped


class Spectrogram(torch.nn.Module):
    def __init__(
        self,
        num_stft_frequencies,
        hop_length=None,
        max_freq_mask=10,
        max_time_mask=45,
        num_freq_mask=2,
        num_time_mask=2,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
    ):
        """
        Creates an object that converts a waveform to and from its spectrogram form by using a
        Short-Time Fourier transform


        Parameters:
        - num_stft_frequencies: How many frequencies to keep during the Short-Time Fourier Transform
        - hop_length: The number of time steps to jump over in the STFT. If your hop length is `l`, you get `audio_length_in_seconds * sample_rate / l` entries in the STFT.
        - max_freq_mask: The maximum number of frequency bins to be masked out. (There are `num_stft_frequencies` bins in total.)
        - max_time_mask: The maximum number of time bins to be masked out. (There are `audio_length_in_seconds * sample_rate /hop_length` time bins in total.)
        - num_freq_mask: The maximum number of frequency masks to be applied.
        - num_time_mask: The maximum number of time masks to be applied.
        # TODO
        - win_length:
        - window:
        - center:
        - pad_mode:
        """
        super(Spectrogram, self).__init__()
        self.max_freq_mask=max_freq_mask
        self.max_time_mask=max_time_mask
        self.num_freq_mask=num_freq_mask
        self.num_time_mask=num_time_mask

        self.n_fft = num_stft_frequencies
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def wav_to_spec(self, wav):
        """
        Computes the Short-Time Fourier Transform of the input audio.

        Mathematically this is equivavlent of the PyTorch STFT which is as follows:

        .. math::
        X[\omega, m] = \sum_{k = 0}^{\text{win\_length-1}}%
                            \text{window}[k]\ \text{input}[m \times \text{hop\_length} + k]\ %
                            \exp\left(- j \frac{2 \pi \cdot \omega k}{\text{n\_fft}}\right),

        @param wav: The input audio waveform.
        """
        return torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode,
            return_complex=False,
        )

    def spec_to_wav(self, stft, wav_length=None):
        return torch.istft(
            stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=wav_length,
            return_complex=False,
        )