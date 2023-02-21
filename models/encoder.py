##################################################
# AUTOENCODER
# Author: Suk Yee Yong
##################################################

import torch
import torch.nn as nn


from torchvision import models
import torch.nn as n

class DenseEncoder(nn.Module):
    def __init__(self, input_shape, num_labels=1, list_linear=[256, 64, 8], **kwargs):
        super(DenseEncoder, self).__init__()
        self.net = [nn.Linear(input_shape[0], list_linear[0]), nn.LeakyReLU()]
        for in_c, out_c in zip(list_linear, list_linear[1:]):
            self.net.extend([
                nn.Linear(in_c, out_c),
                nn.LeakyReLU(),
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


class ConvEncoder(nn.Module):
    def __init__(self, input_shape, num_labels=1, **kwargs):
        super(ConvEncoder, self).__init__()
        # Input shape needs to be [1, nsample]. Add spectrum[np.newaxis, ...]
        list_channel = [16, 32]
        conv_kwargs = {'kernel_size': 50, 'stride': 2}
        self.net = [
            nn.Conv1d(input_shape[0], list_channel[0], **conv_kwargs),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5),
        ]
        for in_c, out_c in zip(list_channel, list_channel[1:]):
            self.net.extend([
                nn.Conv1d(in_c, out_c, **conv_kwargs),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=5),
            ])
        self.net.extend([nn.Flatten()])
        self.net.extend([
            nn.Linear(128, list_channel[-1]),
            nn.ReLU(),
            nn.Linear(list_channel[-1], 8),
            nn.ReLU(),
            nn.Linear(8, num_labels),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

