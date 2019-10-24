import torch
import torch.nn as nn
from src import Conv_2D, RCU, IRU


class Encoder(nn.Module):
    def __init__(self, encoder_cfg):
        super(Encoder, self).__init__()
        assert isinstance(encoder_cfg,dict), TypeError('Expected dict but got {}.'.format(type(encoder_cfg)))
        self.cfg = encoder_cfg
        self.conv = nn.Sequential()
        self._input_conv()
        self._stage_encoder()

    def forward(self, x):
        output = [x]
        for i in range(len(self.conv)):
            output.append(
                self.conv[i](output[-1])
            )
        output = output[2:][::-1]
        return output

    def to_cuda(self):
        self.conv = self.conv.cuda()

    def _input_conv(self):
        self.conv.add_module(
            'input',
            Conv_2D(in_channels=self.cfg['input_channel'],
                    out_channels=self.cfg['stage_channels'][0],
                    kernel_size=self.cfg['input_ker_size'],
                    padding=self.cfg['input_ker_padding'],
                    ))

    def _stage_encoder(self):
        assert self.cfg['num_stage'] > 0
        for i in range(self.cfg['num_stage']):
            if i == 0:
                in_ = self.cfg['stage_channels'][i]
            else:
                in_ = self.cfg['stage_channels'][i-1]
            self.conv.add_module(
                'conv'+str(i),
                nn.Sequential(
                    Conv_2D(
                        in_channels=in_,
                        out_channels=self.cfg['stage_channels'][i],
                        kernel_size=self.cfg['stage_ker_size'][i],
                        padding=self.cfg['stage_ker_padding'][i],
                        bias=False,
                        is_dsc=False,
                        with_bn=True,
                        activation='relu',
                        stride=self.cfg['stage_stide'][i]),
                    Conv_2D(
                        in_channels=self.cfg['stage_channels'][i],
                        out_channels=self.cfg['stage_channels'][i],
                        kernel_size=self.cfg['stage_ker_size'][i],
                        padding=self.cfg['stage_ker_padding'][i]),
                )
            )
