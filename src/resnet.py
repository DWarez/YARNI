import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from encoder import ResNetEncoder
from decoder import ResNetDecoder


class ResNet(nn.Module):
    """ResNet implementation

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output classes
    """
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels, *args, **kwargs)
        self.decoder = ResNetDecoder(in_channels=self.encoder.net[-1].net[-1].expanded_channels, n_classes=n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x