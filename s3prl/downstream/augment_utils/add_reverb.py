# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ add_reverb.py ]
#   Synopsis     [ Add Reverb implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# Add Reverb implementation for SUPERB Author: Kornel Szabo, University of Cambridge

###############
#   IMPORTS   #
###############
# -------------#
import torch
import torchaudio
import einops

# -------------#
from torch import Tensor
from typing import List
from jaxtyping import Float, Int64, jaxtyped
from pathlib import Path


class AddReverb(torch.nn.Module):
    def __init__(
        self,
        dataset_path: str,
        prob: float = 1.0,
    ):
        """
        Creates an object that adds convolutional reverberation to the input signal.
        It uses the MIT Acoustical Reverberation Scene Statistics Survey for the reverberation impulses:
        https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
        """
        super(AddReverb, self).__init__()
        self.dataset_path = Path(dataset_path)
        self.rirs = []
        self.sr = None

        for path in self.dataset_path.iterdir():
            if path.suffix != '.wav':
                continue
            rir, self.sr = torchaudio.load(path)
            rir = rir.squeeze()
            rir = rir / rir.norm()
            self.rirs.append(rir.squeeze())

        self.prob = prob

    def forward(
        self, wavs: List[Float[Tensor, "wav_len"]]
    ) -> List[Float[Tensor, "wav_len"]]:
        """
        Add reverb to the unpadded wavs.
        @param wavs: A list of the padded audio waveforms.
        """
        new_wavs = []
        for i in range(len(wavs)):
            prob = torch.zeros(0).uniform_(0, 1)
            if prob < self.prob:
                wav = wavs[i]
                rir_idx = torch.randint(0, len(self.wavs), (1, ))
                rir = self.rirs[rir_idx]
                new_wav = torchaudio.functional.fftconvolve(wav, rir)
                new_wavs.append(new_wav)
            else:
                new_wavs.append(wav)

        return new_wavs

