"""Testing the AutoPadConv2d module"""
from enum import auto
import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from functools import partial
from auto_pad import AutoPadConv2d


# Defining a layer with kernel size 3
layer_3x3 = partial(AutoPadConv2d, kernel_size=3, bias=False)

auto_pad = layer_3x3(in_channels=32, out_channels=64)

assert auto_pad.in_channels == 32, f"Layer's in_channels {auto_pad.in_channels} instead of 32"
assert auto_pad.out_channels == 64, f"Layer's out_channels {auto_pad.out_channels} instead of 64"
assert auto_pad.kernel_size==(3,3), f"Layer's kernel size {auto_pad.kernel_size} instead of (3,3)" 

print(auto_pad)