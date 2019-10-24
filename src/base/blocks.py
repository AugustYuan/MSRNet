import torch.nn as nn


class IRU(nn.Module):
    def __init__(self, channel):
        super(IRU, self).__init__()
        self.channel=channel
        block1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channel),
        )
        #block1.add_module(nn.ReLU())
        #block1.add_module(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1))
        self.block1 = block1
        block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channel),
        )
        #block2.add_module(nn.ReLU())
        #block2.add_module(name+'_conv2', nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1))
        self.block2 = block2

        block3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channel),
        )
        #block3.add_module(nn.ReLU())
        #block3.add_module(nn.Conv2d(in_channels=self.channel, out_channels=self.channel, kernel_size=3, padding=1))
        self.block3 = block3

    def forward(self, x):
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(x+c2)
        return c1+c3


class RCU(nn.Module):
    def __init__(self, channel):
        super(RCU,self).__init__()
        self.channels = channel
        self.block = nn.Sequential(
             nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False),
             nn.BatchNorm2d(self.channels),
             nn.ReLU(True),
             nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False),
             nn.BatchNorm2d(self.channels),
         )
    def forward(self, x):
        y = self.block(x)
        return nn.ReLU(True)(x+y)


class BF(nn.Module):
    def __init__(self, channel, num_class):
        super(BF, self).__init__()
        self.channels = channel
        self.num = num_class

        self.block1 = nn.Sequential(
            #nn.Conv2d(in_channels=self.num, out_channels=self.channels, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=self.num, out_channels=self.channels, kernel_size=3, padding=1, bias=False),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False),
        )
        self.BN = nn.BatchNorm2d(self.channels)

    def forward(self, pic, fea):
        x1 = self.block1(pic)
        x2 = self.block2(fea)
        y = self.BN(x1+x2)
        return y


class OutU(nn.Module):
    def __init__(self, channel, num_class):
        super(OutU, self).__init__()
        self.channels = channel
        self.num = num_class

        self.block = nn.Sequential(
            nn.BatchNorm2d(self.channels),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=self.channels, out_channels=self.num, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.block(x)
        return x
