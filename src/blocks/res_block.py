"""Residual Block definition"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__(self, in_channels, out_channels)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.block = nn.Identity()
        self.shortcut = nn.Identity()

    
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