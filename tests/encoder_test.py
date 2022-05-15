import sys
sys.path.append('src/blocks')

import torch
import torch.nn as nn
from encoder import ResNetEncoder

encoder = ResNetEncoder()

print(encoder)