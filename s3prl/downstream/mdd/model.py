# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ model.py ]
#   Synopsis     [ the linear model ]
#   Author       [ S3PRL ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import torch
import torch.nn as nn
from jaxtyping import Float, jaxtyped
import typeguard
from torch import Tensor


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(
        self, input_dim, output_class_num, extractor_blocks, extractor_dim, **kwargs
    ):
        super(Model, self).__init__()

        if extractor_blocks > 0:
            self.model = nn.Sequential(
                nn.Linear(input_dim, extractor_dim),
                *[
                    nn.Linear(extractor_dim, extractor_dim)
                    for _ in range(extractor_blocks - 1)
                ],
                nn.LeakyReLU(),
                nn.Linear(extractor_dim, output_class_num)
            )
        # init attributes
        else:
            self.model = nn.Linear(input_dim, output_class_num)

    @jaxtyped
    @typeguard.typechecked
    def forward(
        self, features: Float[Tensor, "batch seq_len upstream_class_num"]
    ) -> Float[Tensor, "batch seq_len downstream_class_num"]:
        preds = self.model(features)
        return preds
