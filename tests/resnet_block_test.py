import sys
sys.path.append('src/blocks')

import torch
import torch.nn
from resnet_block import ResNetBlock

block = ResNetBlock(32, 64)

print(block)