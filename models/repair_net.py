import torch
import torch.nn as nn
from src import IRU, RCU, BF, Conv_2D


base_block={
    'RCU':RCU,
    'IRU':IRU
}

def RepairNets(cfg):
        repairNet = []
        for channel in cfg['channels'][::-1]:
            repairNet.append(
                Repairnet(
                    in_channels=channel,
                    num_class=cfg['num_class'],
                    blocks=cfg['base_block'],
                    is_dsc=cfg['is_dsc']
                )
            )
        return repairNet

class Repairnet(nn.Module):
    def __init__(self,
                 in_channels,
                 num_class,
                 blocks,
                 is_dsc=True):
        super(Repairnet, self).__init__()
        self.in_channels = in_channels
        self.num_class = num_class
        self.blocks = blocks
        self.is_dsc = is_dsc
        self.bf = self._bf_block()
        self.conv = self._conv_block()

    def forward(self, feats, res):
        feats = self.bf(feats, res)
        feats = self.conv(feats)
        return feats


    def _bf_block(self):
        return BF(
            channels=self.in_channels,
            num_class=self.num_class,
            scale_factor=2.0,
            is_dsc=self.is_dsc
        )

    def _conv_block(self):
        conv = nn.Sequential()
        for block in self.blocks:
            conv.add_module(
                block,
                base_block[block](
                    channels=self.in_channels,
                    is_dsc=self.is_dsc)
            )
        conv.add_module(
            'out_conv',
            Conv_2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                is_dsc=False,
                with_bn=False,
                activation=None,
                stride=1
            )
        )
        conv.add_module(
            'out_conv',
            Conv_2D(
                in_channels=self.in_channels,
                out_channels=self.num_class,
                kernel_size=3,
                padding=1,
                bias=False,
                is_dsc=False,
                with_bn=False,
                activation=None,
                stride=1
            )
        )
        return conv

