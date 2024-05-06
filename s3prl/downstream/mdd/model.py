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
from typing import Union

from s3prl.nn import KAN


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(
        self,
        input_dim,
        output_class_num,
        extractor_blocks,
        extractor_dim,
        extractor_type="MLP",
        **kwargs,
    ):
        """
        Initializes the model.
        Supports 3 modes:
        - MLP: A sequence of `extractor_blocks` neural networks of output shape `extractor_dim`
        - KAN: The Komlogorov-Arnold Network from https://arxiv.org/abs/2404.19756
        - FFTKAN: A modification of the KAN using Fourier Series instead of splines.
                  An implementation can be found here: https://github.com/GistNoesis/FourierKAN
        """
        super(Model, self).__init__()

        if extractor_type == "MLP":
            self._init_linear(
                input_dim=input_dim,
                output_class_num=output_class_num,
                extractor_blocks=extractor_blocks,
                extractor_dim=extractor_dim,
            )
        elif extractor_type == "KAN":
            self._init_kan(
                input_dim=input_dim,
                output_class_num=output_class_num,
                extractor_blocks=extractor_blocks,
                extractor_dim=extractor_dim,
            )
        else:
            raise Exception(f"Invalid extractor type {extractor_type}")

    def _init_linear(
        self, input_dim, output_class_num, extractor_blocks, extractor_dim
    ):
        """Initialize an MLP architecture."""
        if extractor_blocks > 0:
            layers = [
                nn.Linear(input_dim, extractor_dim),
                nn.LeakyReLU(),
            ]
            for _ in range(extractor_blocks - 1):
                layers.append(
                    nn.Linear(extractor_dim, extractor_dim),
                )
                layers.append(
                    nn.LeakyReLU(),
                )
            layers.append(
                nn.Linear(extractor_dim, output_class_num),
            )

            self.model = nn.Sequential(*layers)
        # init attributes
        else:
            self.model = nn.Linear(input_dim, output_class_num)

    def _init_kan(self, input_dim, output_class_num, extractor_blocks, extractor_dim):
        """Initialize a KAN architecture."""
        if extractor_blocks > 0:
            self.model = KAN(
                [input_dim]
                + [extractor_dim for i in range(extractor_blocks - 1)]
                + [output_class_num],
                # TODO: change this without breaking abstractions
                # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
        else:
            self.model = KAN([input_dim, output_class_num])

    def _init_fftkan(
        self, input_dim, output_class_num, extractor_blocks, extractor_dim
    ):
        raise NotImplementedError

    @jaxtyped
    @typeguard.typechecked
    def forward(
        self,
        features: Union[
            Float[Tensor, "batch seq_len upstream_class_num"],
            Float[Tensor, "batch_and_len upstream_class_num"],
        ],
    ) -> Union[
        Float[Tensor, "batch seq_len downstream_class_num"],
        Float[Tensor, "batch_and_len downstream_class_num"],
    ]:
        preds = self.model(features)
        return preds
