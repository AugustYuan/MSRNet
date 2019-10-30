from .Conv_2D import Conv_2D
import torch.nn as nn


class IRU(nn.Module):
    def __init__(self,
                 channels,
                 kernel_size=3,
                 padding=1,
                 bias=False,
                 is_dsc=False,
                 with_bn=True,
                 stride=1):
        super(IRU, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.is_dsc = is_dsc
        self.stride = stride
        self.conv1 = Conv_2D(self.channels,
                    self.channels,
                    self.kernel_size,
                    self.padding,
                    self.bias,
                    self.is_dsc,
                    self.with_bn,
                    self.stride)
        self.conv2 = Conv_2D(self.channels,
                    self.channels,
                    self.kernel_size,
                    self.padding,
                    self.bias,
                    self.is_dsc,
                    self.with_bn,
                    None)
        self.conv3 = Conv_2D(self.channels,
                    self.channels,
                    self.kernel_size,
                    self.padding,
                    self.bias,
                    self.is_dsc,
                    self.with_bn,
                    None)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(nn.ReLU()(x+x2))
        return nn.ReLU()(x1+x3)
