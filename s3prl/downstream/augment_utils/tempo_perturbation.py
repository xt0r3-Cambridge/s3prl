# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ tempo_perturbation.py ]
#   Synopsis     [ Tempo Perturbation implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Tempo Perturbation implementation for SUPERB Author: Kornel Szabo, University of Cambridge
# source; https://www.mdpi.com/2076-3417/6/2/57
# implementation guided by: https://github.com/KAIST-MACLab/PyTSMod/blob/main/pytsmod/wsolatsm.py 

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
import torch.nn.functional as F


class TempoPerturbation(torch.nn.Module):
    def __init__(
        self,
        prob: float = 1.0,
        alpha_low: float = 0.5,
        alpha_high: float = 1.2,
        sample_rate=16000,
        delta_max: int = 512,
        N: int = 2048,
        window: str = "hann",
    ):
        """
        Creates an object that adds Gaussian noise to the input signal.
        to any input.

        prob: the probability of the augmentation occurring
        alpha_low: the lowest possible ratio between the augmented length and
                   the orignal length of an audio waveform
        alpha_high: the highest possible ratio between the augmented length and
                   the orignal length of an audio waveform
        """
        super(TempoPerturbation, self).__init__()
        self.valid_windows = ["bartlett", "blackman", "hamming", "hann", "kaiser"]

        self.delta_max = delta_max
        self.N = N
        self.Hs = self.N // 2

        self.sr = sample_rate

        self.prob = prob
        self._validate_prob()

        self.alpha_low = alpha_low
        self.alpha_high = alpha_high
        self._validate_alpha()

        self._validate_window(window)

    def _validate_prob(self):
        assert (
            0 <= self.prob <= 1
        ), f"Perturbation probability needs to be in [0, 1]. Found: {self.prob}"

    def _validate_alpha(self):
        assert (
            0 <= self.alpha_low <= self.alpha_high
        ), f"""Invalid alpha values: {self.alpha_low}, {self.alpha_high}
The following needs to hold: 0 <= alpha_low <= alpha_high
"""

    def _validate_window(self, window):
        assert (
            window in self.valid_windows
        ), f"""Invalid window type: {window}
Currently only the following windows are supported:
{self.valid_windows}
"""
        if window == "hann":
            self.window = torch.hann_window(self.N)
        elif window == "bartlett":
            self.window = torch.bartlett_window(self.N)
        elif window == "blackman":
            self.window = torch.blackman_window(self.N)
        elif window == "hamming":
            self.window = torch.hamming_window(self.N)
        elif window == "kaiser":
            self.window = torch.kaiser_window(self.N)

    def cross_correlate(self, x: Float[Tensor, "x_len"], y: Float[Tensor, "x_len"]):
        if x.size() < y.size():
            return F.conv1d(y[None, None], x[None, None]).squeeze().flip(0)

        return F.conv1d(x[None, None], y[None, None]).squeeze()

    def forward(
        self, wavs: List[Float[Tensor, "wav_len"]]
    ) -> List[Float[Tensor, "wav_len"]]:
        """
        Computes the tempo perturbation of the input audio.
        @param wavs: A list of the unpadded audio waveforms.
        """
        self.window = self.window.to(wavs[0].device)
        new_wavs = []
        for i in range(len(wavs)):
            wav = wavs[i]
            p = torch.zeros(1, device=wav.device).uniform_(0, 1)

            if p < self.prob:
                alpha = torch.zeros(1, device=wav.device).uniform_(
                    self.alpha_low, self.alpha_high
                )


                new_len = int(torch.ceil(alpha * wav.shape[-1]))

                # print(f"{new_len=}")

                Hs_boundaries = torch.arange(0, new_len + self.N // 2, self.Hs, device=wav.device)

                # TODO: maybe make this more exact with linear interpolation
                Ha_boundaries = (Hs_boundaries / alpha).round().int()

                Ha_hops = torch.diff(Ha_boundaries, prepend=torch.zeros(1, device=wav.device))


                min_Ha_hop_frac = (self.Hs / Ha_hops).min()

                left_pad = self.N // 2 + self.delta_max
                right_pad = (torch.ceil(1 / min_Ha_hop_frac) * self.N + self.delta_max).int().item()

                wav_pad = F.pad(wav, (left_pad, right_pad), "constant", 0)

                # The maximal points in the original signal where the boundaries of the 
                # adjusted signal can be
                Ha_boundaries += self.delta_max

                new_wav = torch.zeros(new_len + 2 * self.N, device=wav.device)
                win_sum = torch.zeros(new_len + 2 * self.N, device=wav.device)

                delta = 0
                for i in range(len(Ha_boundaries)):
                    # This is the block corresponding to x_m in the original paper
                    adj_wav_chunk = wav_pad[Ha_boundaries[i] + delta: Ha_boundaries[i] + delta + self.N]
                    new_wav[Hs_boundaries[i]:Hs_boundaries[i]+self.N] += adj_wav_chunk * self.window
                    win_sum[Hs_boundaries[i]:Hs_boundaries[i]+self.N] += self.window

                    if i < len(Ha_boundaries) - 1:
                        # Step one original block length (Hs)
                        Hs_next_chunk = wav_pad[Ha_boundaries[i] + delta + self.Hs: Ha_boundaries[i] + delta + self.Hs + self.N]

                        Ha_next_chunk_range = wav_pad[Ha_boundaries[i+1] - self.delta_max: Ha_boundaries[i+1] + self.N + self.delta_max]

                        cross_corr = self.cross_correlate(Hs_next_chunk, Ha_next_chunk_range)
                        # print(f"{cross_corr=}")
                        # print(f"{cross_corr.shape=}")
                        max_index = cross_corr.argmax()

                        delta = self.delta_max - max_index
                
                # Ill-behaved windows should now explode the result
                win_sum[win_sum < 1e-3] = 1

                # Normalisation by the sums of windows at every point
                new_wav /= win_sum

                # Remove padding
                new_wav = new_wav[self.N // 2: new_len + self.N // 2]
            
                new_wavs.append(new_wav)
            else:
                new_wavs.append(wav)

        return new_wavs

    # def forward(
    #     self, wavs: Float[Tensor, "wav_cnt wav_len"], wav_lens: Int64[Tensor, "wav_cnt"]
    # ) -> List[Float[Tensor, "wav_cnt wav_len"]]:
    #     """
    #     Change the tempo of the padded wavs.
    #     @param wavs: A list of the padded audio waveforms.
    #     """
    #     # Don't add noise when with an (1-self.prob) probability
    #     no_perturb_prob = torch.distributions.bernoulli.Bernoulli(
    #         torch.ones(wavs.size()[0], device=self.device) * (1 - self.prob)
    #     )
    #     prob_mask = no_perturb_prob.sample().bool()

    #     # Don't add noise to the padding
    #     len_mask = torch.arange(wavs.shape[-1], device=self.device).expand(
    #         len(wav_lens), wavs.shape[-1]
    #     ) > wav_lens.unsqueeze(1)

    #     # Unify the masks
    #     mask = len_mask
    #     mask[prob_mask, :] = True

    #     # Generate noise for the wavs:
    #     # The noise comes from sampling an N(0, Sigma), where Sigma is wav_cnt * wav_cnt diagonal
    #     # and randomly generated between snr_low and snr_high.
    #     # We sample it wav_len times and that will be the initial noise
    #     std = (
    #         torch.ones(wavs.size()[0], device=self.device)
    #         .uniform_(self.snr_low, self.snr_high, generator=self.rng)
    #         .unsqueeze(-1)
    #         .expand(-1, wavs.size()[-1])
    #     )
    #     noise = torch.normal(
    #         0,
    #         std,
    #     )

    #     # The noise is scaled by the amplitude of the input audio
    #     noise_with_amp = wavs * noise
    #     noise_with_amp[mask] = 0

    #     new_wavs = wavs + noise_with_amp
    #     return new_wavs
