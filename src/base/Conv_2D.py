import torch.nn as nn


class Conv_2D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 bias=False,
                 is_dsc=False,
                 with_bn=True,
                 activation='relu',
                 stride=1):
        super(Conv_2D, self).__init__()
        assert isinstance(in_channels, int) and in_channels > 0, \
            TypeError('Expected int but got {}.'.format(type(in_channels)))
        assert isinstance(out_channels, int) and out_channels > 0, \
            TypeError('Expected int but got {}.'.format(type(out_channels)))
        assert isinstance(kernel_size, int) and kernel_size > 0, \
            TypeError('Expected int but got {}.'.format(type(kernel_size)))
        assert isinstance(padding, int) and padding >= 0, \
            ValueError('The value of key "padding" is not expected.')
        assert isinstance(stride, int) and stride >= 1, \
            TypeError('The value of key "stride" is not expected.')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.activation = activation
        self.stride = stride
        if is_dsc:
            self.conv2d = self._dsc()
        else:
            self.conv2d = self._conv()

    def forward(self, x):
        x = self.conv2d(x)
        if self.activation == 'relu':
            x = nn.ReLU()(x)
        elif self.activation == 'sigmod':
            x = nn.Sigmoid()(x)
        return x

    def _dsc(self):
        dcn = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.in_channels,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      stride=self.stride,
                      bias=False,
                      groups=self.in_channels),
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=1,
                      padding=0,
                      stride=self.stride,
                      bias=self.bias),
        )
        if self.with_bn:
            dcn.add_module('bn', nn.BatchNorm2d(self.out_channels))
        return dcn

    def _conv(self):
        conv = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.out_channels,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias,
                      stride=self.stride),
        )
        if self.with_bn:
            conv.add_module('bn', nn.BatchNorm2d(self.out_channels))
        return conv