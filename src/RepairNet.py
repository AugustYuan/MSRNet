import torch.nn as nn
from blocks import *

class RN(nn.Module):
    def __init__(self, channel, num_class):
        super(RN, self).__init__()
        self.channels = channel
        self.num_class = num_class

        self.bf = BF(self.channels, self.num_class)
        self.block = nn.Sequential(
            RCU(self.channels),
            IRU(self.channels),
            OutU(self.channels, self.num_class),
        )

    def forward(self, pic, fea):
        x = self.bf(pic, fea)
        x = self.block(x)
        return x