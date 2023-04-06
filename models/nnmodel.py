##################################################
# AUTOENCODER
# Author: Suk Yee Yong
##################################################

import torch
import torch.nn as nn


from torchvision import models
import torch.nn as n

class DenseEncoder(nn.Module):
    """Dense encoder for supervised learning"""
    def __init__(self, input_shape, num_labels=1, list_linear=[64, 64], dropout=0.1, **kwargs):
        super(DenseEncoder, self).__init__()
        list_linear = [input_shape[0], *list_linear]
        self.net = []
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


class Autoencoder(nn.Module):
    """Autoencoder for unsupervised learning"""
    def __init__(self, input_shape, list_linear=[512, 256], dropout=0.1, **kwargs):
        super(Autoencoder, self).__init__()
        list_linear = [input_shape[0], *list_linear]
        # Encoder
        enc = []
        for in_c, out_c in zip(list_linear[:-2], list_linear[1:-1]):
            enc.extend([
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        enc.extend([nn.Linear(list_linear[-2], list_linear[-1])])
        self.encoder = nn.Sequential(*enc)
        # Decoder
        dec = []
        list_linear.reverse()
        for in_c, out_c in zip(list_linear[:-2], list_linear[1:-1]):
            dec.extend([
                nn.Linear(in_c, out_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        dec.extend([
            nn.Linear(list_linear[-2], list_linear[-1]),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*enc, *dec)
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

