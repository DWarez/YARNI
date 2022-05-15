import sys
sys.path.append('src/blocks')
sys.path.append('src')

import torch
import torch.nn as nn

from resnet import ResNet

model = ResNet(3, 10)
_tensor = torch.ones([1, 3, 128, 128])

print(model)
print(model(_tensor).shape)