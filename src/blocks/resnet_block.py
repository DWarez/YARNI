"""ResNet block definition"""

import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from collections import OrderedDict
from res_block import ResidualBlock

from functools import partial
from auto_pad import AutoPadConv2d

# In the original paper, the convolution is composed by 3x3 kernels
layer_3x3 = partial(AutoPadConv2d, kernel_size=3, bias=False)


class ResNetBlock(ResidualBlock):
    """Class of ResNet's Residual Block

    Args:
        in_channles (int): number of input channels
        out_channels (int): number of output channels
        expansion (int): expansion coefficient for the output channels
        downsampling (int): size of the stride used in the 2D convolution when the shortucut is applied
        conv (f): convolution function used when creating a convolution+batch normalization block
    """
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=layer_3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv

        self.shortcut = nn.Sequential(OrderedDict({
            "convolution": nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            "batch_normalization": nn.BatchNorm2d(self.expanded_channels)
        })) if super().apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion