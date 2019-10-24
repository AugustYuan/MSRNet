from src import IRU, RCU, Conv_2D
import torch.nn as nn

blocks = {
    "RCU":RCU,
    "IRU":IRU
}


class SSDecoder(nn.Module):
    def __init__(self, cfg):
        super(SSDecoder, self).__init__()
        assert isinstance(cfg, dict)
        self.cfg = cfg
        self.conv = self._build_conv()

    def forward(self, feat):
        feat = self.conv(feat)
        return feat

    def to_cuda(self):
        self.conv = self.conv.cuda()

    def _build_conv(self):
        conv = nn.Sequential()
        for block in self.cfg['base_block']:
            conv.add_module(
                block,
                blocks[block](
                    channels=self.cfg['channels'],
                    is_dsc=self.cfg['is_dsc'])
            )
        conv.add_module(
            'out_conv',
            Conv_2D(
                in_channels=self.cfg['channels'],
                out_channels=self.cfg['num_class'],
                kernel_size=1,
                padding=0,
                bias=False,
                is_dsc=self.cfg['is_dsc'],
                with_bn=False,
                activation=None,
                stride=1
            )
        )
        return conv