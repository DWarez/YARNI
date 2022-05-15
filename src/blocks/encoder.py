import sys
sys.path.append('src/blocks')

import torch.nn as nn
from resnet_block import ResNetBlock
from resnet_layer import ResNetLayer


class ResNetEncoder(nn.Module):
    """ResNetEncoder

    Args:
        in_channels (int): number of input channels
        blocks_sizes (list): list of output channels per block
        depths (list): list of number of layers per block
        activation (nn.Module): activation function
        base (nn.Module): baseline block
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], depths=[2, 2, 2, 2], activation=nn.ReLU, base=ResNetBlock, *args, **kwargs):
        super().__init__()

        self.blocks_sizes, self.depths, self.activation, self.base = blocks_sizes, depths, activation, base

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            self.activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.in_out_block_sizes = list(zip(self.blocks_sizes, self.blocks_sizes[1:]))

        self.net = nn.Sequential(
            ResNetLayer(self.blocks_sizes[0], self.blocks_sizes[0], base=self.base, n_layers=self.depths[0], activation=self.activation, *args, **kwargs),
            *[ResNetLayer(in_channels * base.expansion, 
                          out_channels, base=base, n_layers=n, activation=activation, 
                          *args, **kwargs) for (in_channels, out_channels), n in zip(self.in_out_block_sizes, self.depths[1:])]  
        )


    def forward(self, x):
        x = self.gate(x)
        for block in self.net:
            x = block(x)
        return x