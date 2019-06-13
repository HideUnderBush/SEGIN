import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import os.path as osp

class FeatureExtractor(nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]

