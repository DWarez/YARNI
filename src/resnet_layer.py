import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from resnet_block import ResNetBlock


class ResNetLayer(nn.Module):
    """ResNet Layer

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, base=ResNetBlock, n_layers=1, *args, **kwargs):
        super().__init__()

        downsampling = 2 if in_channels != out_channels else 1
        self.net = nn.Sequential(
            base(in_channels, out_channels, downsampling=downsampling, *args, **kwargs),
            *[base(out_channels * base.expansion, out_channels, downsampling=1, *args, **kwargs) for _ in range(n_layers-1)]
            #*[base(out_channels * 1, out_channels, downsampling=1, *args, **kwargs) for _ in range(n_layers-1)]
        )


    def forward(self, x):
        x = self.net(x)
        return x