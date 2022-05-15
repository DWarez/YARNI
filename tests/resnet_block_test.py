import sys
sys.path.append('src/blocks')

import torch
import torch.nn
from resnet_block import ResNetBlock

_tensor = torch.ones((1, 32, 224, 224))
block = ResNetBlock(32, 64)

print(block)
print(block(_tensor).shape)