import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from resnet_layer import ResNetLayer
from resnet_block import ResNetBlock

_tensor = torch.ones((1, 64, 48, 48))

layer = ResNetLayer(64, 128, base=ResNetBlock, n_layers=3)

print(layer)
print(layer(_tensor).shape)