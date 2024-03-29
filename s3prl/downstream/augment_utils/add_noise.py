# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ add_noise.py ]
#   Synopsis     [ Add Noise implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Add Noise implementation for SUPERB Author: Kornel Szabo, University of Cambridge

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


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        device: str,
        prob: float = 1.0,
        snr_low: float = 0.0,
        snr_high: float = 0.1,
        seed: int = 1337,
    ):
        """
        Creates an object that adds Gaussian noise to the input signal.
        to any input.
        """
        super(AddNoise, self).__init__()
        self.device = device
        self.seed = seed
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        self.prob = prob
        self.snr_low = snr_low
        self.snr_high = snr_high

    def forward(
        self, wavs: Float[Tensor, "wav_cnt wav_len"], wav_lens: Int64[Tensor, "wav_cnt"]
    ) -> List[Float[Tensor, "wav_cnt wav_len"]]:
        """
        Add noise to the padded wavs.
        @param wavs: A list of the padded audio waveforms.
        """
        # Don't add noise when with an (1-self.prob) probability
        no_noise_prob = torch.distributions.bernoulli.Bernoulli(
            torch.ones(wavs.size()[0], device=self.device) * (1 - self.prob)
        )
        prob_mask = no_noise_prob.sample().bool()

        # Don't add noise to the padding
        len_mask = torch.arange(wavs.shape[-1], device=self.device).expand(
            len(wav_lens), wavs.shape[-1]
        ) > wav_lens.unsqueeze(1)

        # Unify the masks
        mask = len_mask
        mask[prob_mask, :] = True

        # Generate noise for the wavs:
        # The noise comes from sampling an N(0, Sigma), where Sigma is wav_cnt * wav_cnt diagonal
        # and randomly generated between snr_low and snr_high.
        # We sample it wav_len times and that will be the initial noise
        std = (
            torch.ones(wavs.size()[0], device=self.device)
            .uniform_(self.snr_low, self.snr_high, generator=self.rng)
            .unsqueeze(-1)
            .expand(-1, wavs.size()[-1])
        )
        noise = torch.normal(
            0,
            std,
        )

        # The noise is scaled by the amplitude of the input audio
        noise_with_amp = wavs * noise
        noise_with_amp[mask] = 0

        new_wavs = wavs + noise_with_amp
        return new_wavs
