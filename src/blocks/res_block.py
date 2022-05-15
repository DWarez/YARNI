"""Residual Block definition"""

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Abstract class of a ResidualBlock, which is used as a base for the ResNet's Residual Block

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block = nn.Identity()
        self.shortcut = nn.Identity()

    
    """In the forward step we compute H(x) + x, when using shortcuts"""
    def forward(self, x):
        residual = x
        if self.apply_shortcut:
            residual = self.shortcut(x)
        x = self.block(x)
        x += residual
        return x


    @property
    def apply_shortcut(self):
        return self.in_channels != self.out_channels