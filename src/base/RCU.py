import torch.nn as nn
from .Conv_2D import Conv_2D


class RCU(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size=3,
                 padding=1,
                 bias=False,
                 is_dsc=False,
                 with_bn=True,
                 stride=1):
        super(RCU, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.is_dsc = is_dsc
        self.stride = stride
        self.conv = self._conv_block()

    def forward(self, x):
        y = self.conv(x)
        return nn.ReLU()(x+y)

    def _conv_block(self):
        conv = nn.Sequential(
            Conv_2D(in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                    is_dsc=self.is_dsc,
                    with_bn=self.with_bn,
                    stride=self.stride),
            Conv_2D(in_channels=self.channels,
                    out_channels=self.channels,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    bias=self.bias,
                    is_dsc=self.is_dsc,
                    with_bn=self.with_bn,
                    activation=None),
        )
        return conv