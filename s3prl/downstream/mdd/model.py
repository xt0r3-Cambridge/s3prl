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
from torch.nn.functional import log_softmax


#########
# MODEL #
#########
class Model(nn.Module):
    def __init__(self, input_dim, output_class_num, **kwargs):
        super(Model, self).__init__()
        
        # init attributes
        self.linear = nn.Linear(input_dim, output_class_num)          


    def forward(self, features):
        preds = self.linear(features)
        log_preds = log_softmax(preds)
        return log_preds
