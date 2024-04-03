# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ vltp.py ]
#   Synopsis     [ Vocal-Tract Length Perturbation implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Vocal-Tract Length Perturbation implementation for SUPERB Author: Kornel Szabo, University of Cambridge
# Source: https://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf#page=5&zoom=100,0,0

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


class VLTP(torch.nn.Module):
    def __init__(
        self,
        f_hi=4800,
        sample_rate=16000,
        prob=1.0,
        noise_low=0.8,
        noise_high=1.2,
        num_stft_frequencies=2048,
        hop_length=None,
        win_length=None,
        window=None,
        center=True,
        pad_mode="reflect",
    ):
        """
        Creates an object that performs [vocal-tract length perturbation](https://www.cs.toronto.edu/~ndjaitly/jaitly-icml13.pdf#page=5&zoom=100,0,0)
        to any input.

        The workflow is as follows:
        1. Compute the [Short-Time Fourier Transform](https://pytorch.org/docs/stable/_modules/torch/functional.html#stft) of the input.
            The shape of this is `[num_stft_frequencies x (audio_length / hop_length)]`
        2. Multiply the phase of the Fourier transform by some Gaussian noise.
        3. Mask out some frequency ranges of the Fourier transform
        4. Mask out some time ranges of the Fourier transform

        Parameters:
        - f_hi: the boundary frequency where the multiplicative shift turns into an additive shift
        - sample_rate: the sample rate of your input audio
        - prob: the probability of the perturbation
        - noise_low: the lowest possible multiplicative noise factor for the augmentation
        - noise_high: the highest possible multiplicative noise factor for the augmentation
        - num_stft_frequencies: How many frequencies to keep during the Short-Time Fourier Transform
        - hop_length: The number of time steps to jump over in the STFT. If your hop length is `l`, you get `audio_length_in_seconds * sample_rate / l` entries in the STFT.
        # TODO
        - win_length:
        - window:
        - center:
        - pad_mode:
        """
        super(VLTP, self).__init__()

        self.f_hi = f_hi
        self.sr = sample_rate
        self.prob = prob
        self.noise_low = noise_low
        self.noise_high = noise_high
        self.n_fft = num_stft_frequencies
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def apply_stft(self, wav):
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
            return_complex=True,
        )

    def invert_stft(self, stft, wav_length=None):
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

    def to_index(self, freq):
        """
        Converts a frequency to an index into an STFT array with a sample rate
        of `self.sr` and a n_fft of `self.n_fft`.
        """
        return freq * self.n_fft / self.sr

    def forward(
        self, wavs: List[Float[Tensor, "wav_len"]]
    ) -> List[Float[Tensor, "wav_len"]]:
        """
        Computes the vocal-tract length perturbation of the input audio.
        @param wavs: A list of the unpadded audio waveforms.
        """
        new_wavs = []
        for i in range(len(wavs)):
            wav = wavs[i]
            p = torch.zeros(1, device=wav.device).uniform_(
                0, 1
            )
            if p < self.prob:
                alpha = torch.zeros(1, device=wav.device).uniform_(
                    self.noise_low, self.noise_high
                )
                f_hi = (self.f_hi * min(alpha, 1) / alpha)
                f_hi_idx = self.to_index(f_hi).long()

                stft = self.apply_stft(wav)
                # These are the frequencies the individual indices correspond to
                new_freqs = torch.arange(0, stft.shape[0], device=wav.device) * self.sr / self.n_fft
                # We convert them to their new frequencies
                new_freqs[:f_hi_idx] = new_freqs[:f_hi_idx] * alpha
                new_freqs[f_hi_idx:] = (new_freqs[f_hi_idx:] - self.sr / 2) * (
                    self.sr / 2 - self.f_hi * min(alpha, 1)
                ) / (self.sr / 2 - self.f_hi * min(alpha, 1) / alpha) + self.sr / 2
                # We convert the new frequencies back to indices in the stft
                new_freqs = (new_freqs * self.n_fft / self.sr).long()

                new_wavs.append(self.invert_stft(stft[new_freqs], wav.shape[0]))
            else:
                new_wavs.append(wav)

        return new_wavs
