import torch
import torch.nn as nn


"""The AutoPadConv2d module implements auto padding on a Conv2d module"""
class AutoPadConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super.__init__(self, *args, **kwargs)
        self.padding = (self.kernel_size[0]//2, self.kernel_size[1]//2)
