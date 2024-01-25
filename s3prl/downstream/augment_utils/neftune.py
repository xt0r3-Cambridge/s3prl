# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ neftune.py ]
#   Synopsis     [ NEFTune implementation for S3PRL ]
#   Author       [ S3PRL, Kornel Szabo ]
#   Copyright    [ Copyleft(c) ]
"""*********************************************************************************************"""
# NEFTune implementation for SUPERB Author: Kornel Szabo, University of Cambridge

###############
#   IMPORTS   #
###############
#-------------#
import torch
#-------------#
from torch import Tensor
from typing import List
from jaxtyping import Float, Int64, jaxtyped

class NEFTune(torch.nn.Module):
    def __init__(self, noise_alpha=0.1):
        super(NEFTune, self).__init__()
        self.noise_alpha=noise_alpha

    def forward(self, features: List[Float[Tensor, "seq_len output_dim"]]) -> List[Float[Tensor, "seq_len output_dim"]]:
        for i in range(len(features)):
            batch = features[i]
            dims = torch.tensor(batch.size(0) * batch.size(1))
            mag_norm = self.noise_alpha / torch.sqrt(dims)
            batch = batch + torch.zeros_like(batch).uniform_(-mag_norm, mag_norm)
            features[i] = batch
        return features