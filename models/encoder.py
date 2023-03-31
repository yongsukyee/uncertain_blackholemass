##################################################
# AUTOENCODER
# Author: Suk Yee Yong
##################################################

import torch
import torch.nn as nn


from torchvision import models
import torch.nn as n

class DenseEncoder(nn.Module):
    def __init__(self, input_shape, num_labels=1, list_linear=[64, 64], dropout=0.1, **kwargs):
        super(DenseEncoder, self).__init__()
        self.net = [nn.Linear(input_shape[0], list_linear[0]), nn.ReLU()]
        for in_c, out_c in zip(list_linear, list_linear[1:]):
            self.net.extend([
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        self.net.extend([
            nn.Linear(list_linear[-1], num_labels),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

