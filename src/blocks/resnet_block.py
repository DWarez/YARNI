import sys
sys.path.append('src/blocks')

import torch.nn as nn
from collections import OrderedDict
from res_block import ResidualBlock

from functools import partial
from auto_pad import AutoPadConv2d

# In the original paper, the convolution is composed by 3x3 kernels
layer_3x3 = partial(AutoPadConv2d, kernel_size=3, bias=False)


class ResNetBlock(ResidualBlock):
    """ResNet's Basic Block
    It's a concrete implementation of the abstract ResidualBlock, where we apply downsampling
    via 2D convolution + batch normalization in the shortcut, while also expanding the
    dimension of the output channels, if needed.
    The main block (a.k.a. the CNN of the ResNet's block) applyes in a sequence 
    convolution -> batch normalization -> non linearity -> convolution -> batch normalization

    Args:
        in_channles (int): number of input channels
        out_channels (int): number of output channels
        downsampling (int): size of the stride used in the 2D convolution when the shortucut is applied
        conv (f): convolution function used when creating a convolution+batch normalization block
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, activation=nn.ReLU, downsampling=1, conv=layer_3x3, *args, **kwargs):
        super().__init__(in_channels, out_channels)
        self.downsampling, self.conv = downsampling, conv

        self.block = nn.Sequential(OrderedDict({
                "conv1": self.conv(in_channels, out_channels, stride=self.downsampling, bias=False, *args, **kwargs),
                "batch_norm1": nn.BatchNorm2d(out_channels),
                "activation": activation(),
                "conv2": self.conv(out_channels, self.expanded_channels, *args, **kwargs),
                "batch_norm2": nn.BatchNorm2d(out_channels) 
            }))

        self.shortcut = nn.Sequential(OrderedDict({
            "convolution": nn.Conv2d(self.in_channels, self.expanded_channels, kernel_size=1, stride=self.downsampling, bias=False),
            "batch_normalization": nn.BatchNorm2d(self.expanded_channels)
        })) if super().apply_shortcut else None


    @property
    def expanded_channels(self):
        return self.out_channels * ResNetBlock.expansion