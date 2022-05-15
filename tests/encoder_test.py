import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from encoder import ResNetEncoder

encoder = ResNetEncoder()
_tensor = torch.ones((1, 3, 48, 48))

print(encoder)
print(encoder(_tensor).shape)