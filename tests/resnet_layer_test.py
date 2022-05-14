import sys
sys.path.append('src')

import torch
import torch.nn as nn
from resnet_layer import ResNetLayer
from resnet_block import ResNetBlock


layer = ResNetLayer(64, 128, base=ResNetBlock, n_layers=3)
print(layer)