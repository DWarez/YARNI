import sys
sys.path.append('src')

import torch
import torch.nn as nn
from resnet_layer import ResNetLayer
from resnet_block import ResNetBlock
from encoder import ResNetEncoder

encoder = ResNetEncoder()

print(encoder)