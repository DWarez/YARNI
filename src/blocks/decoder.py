import torch.nn as nn

class ResNetDecoder(nn.Module):
    """ResNet Decoder
    Applies 2D average pooling and passes the result to a linear classifier

    Args:
        in_channels (int): number of input channels
        n_classes (int): number of output classes
    """
    def __init__(self, in_channels, n_classes):
        super().__init__()

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.decoder = nn.Linear(in_channels, n_classes)


    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x