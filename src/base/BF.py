from .Conv_2D import Conv_2D
import torch.nn as nn


class BF(nn.Module):
    def __init__(self,
                 channels,
                 num_class,
                 scale_factor=2.0,
                 kernel_size1=3,
                 padding1=1,
                 kernel_size2=3,
                 padding2=1,
                 is_dsc=True,
                 bias=False,
                 with_bn=True,
                 sample='bilinear'):
        super(BF, self).__init__()
        assert isinstance(scale_factor, float), TypeError('Expect float but got {}.'.format(type(scale_factor)))
        self.channels = channels
        self.num_class = num_class
        self.scale_factor = scale_factor
        self.kernel_size1 = kernel_size1
        self.padding1 = padding1
        self.kernel_size2 = kernel_size2
        self.padding2 = padding2
        self.bias = bias
        self.is_dsc=is_dsc
        self.with_bn = with_bn
        self.sample = sample

        self.upsample, self.conv = self._conv_block()

        if with_bn:
            self.BN = nn.BatchNorm2d(self.channels)

    def forward(self, feat, res):
        res = self.upsample(res)
        feat = self.conv(feat) + res
        if self.with_bn:
            feat = self.BN(feat)
        return feat

    def _conv_block(self):
        upsam = nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode=self.sample),
            Conv_2D(self.num_class,
                    self.channels,
                    self.kernel_size1,
                    self.padding1,
                    self.bias,
                    self.is_dsc,
                    False,
                    None)
        )
        merge = nn.Sequential(
            Conv_2D(self.channels,
                    self.channels,
                    self.kernel_size2,
                    self.padding2,
                    self.bias,
                    self.is_dsc,
                    False,
                    None)
        )
        return upsam, merge
