# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ phase_perturbation.py ]
#   Synopsis     [ Time-Domain Spec Augmentation implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Time-Domain Spec Augmentation implementation for SUPERB Author: Kornel Szabo, University of Cambridge
# Source: https://arxiv.org/pdf/2312.08571.pdf

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


class TimeDomainSpecAug(torch.nn.Module):
    def __init__(
        self,
        noise_scale,
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
        A time-domain approximation of the Spectral Augmentation algorithm.

        The workflow is as follows:
        1. Compute the [Short-Time Fourier Transform](https://pytorch.org/docs/stable/_modules/torch/functional.html#stft) of the input.
            The shape of this is `[num_stft_frequencies x (audio_length / hop_length)]`
        2. Multiply the phase of the Fourier transform by some Gaussian noise.
        3. Mask out some frequency ranges of the Fourier transform
        4. Mask out some time ranges of the Fourier transform

        Parameters:
        - noise_scale: How much noise to add to the phase of the audio. The noise comes from a N(1, noise_scale) distribution
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
        super(TimeDomainSpecAug, self).__init__()
        self.max_freq_mask=max_freq_mask
        self.max_time_mask=max_time_mask
        self.num_freq_mask=num_freq_mask
        self.num_time_mask=num_time_mask

        self.noise_scale = noise_scale
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

    def compute_theta(self, stft):
        """
        Computes the phase spectrum matrix of the Short-Time Fourier Trasform.
        The audio in the transformed form is considered as a complex number
        amp * (cos(phase) + i * sin(phase))
        @param stft: the Short-Time Fourier Transform of the input audio
        """
        return torch.angle(stft)

    def invert_phase_matrix(self, theta, amp):
        """
        Computes the inverse of the phase spectrum matrix theta to
        recover the Short-Term Fourier Transform stft given the
        amplitude information amp.

        By definition, `stft =  amp * (cos(theta) + i * sin(theta))`
        """
        real = torch.cos(theta) * amp
        imag = torch.sin(theta) * amp
        return torch.complex(real, imag)

    def randomise_phase(self, theta):
        """
        Randomly permutes the phase spectrum matrix.
        @param theta: The phase spectrum matrix

        According to the paper if we just randomly multiply every element by a
        Gaussian, we get too much noise, so we instead multiply everything in
        the same column by the same random number.
        """
        noise = torch.empty(theta.shape[-1], device=theta.device).normal_(1, self.noise_scale)
        noise = einops.repeat(noise, "y -> x y", x=theta.shape[0])
        return noise * theta

    def forward(
        self, wavs: List[Float[Tensor, "wav_len"]]
    ) -> List[Float[Tensor, "wav_len"]]:
        """
        Computes the Time-Domain Spec Augmentation of the input audio.
        @param wavs: A list of the unpadded audio waveforms.
        """
        new_wavs = []
        for i in range(len(wavs)):
            wav = wavs[i]
            stft = self.apply_stft(wav)
            amp = stft.abs()
            theta = self.compute_theta(stft)
            random_theta = self.randomise_phase(theta)
            random_stft = self.invert_phase_matrix(random_theta, amp)
            new_wavs.append(self.invert_stft(random_stft, wav.shape[0]))

        return new_wavs
