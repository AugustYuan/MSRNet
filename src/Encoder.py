import torch.nn as nn
from blocks import RCU

class Encoder(nn.Module):
    def __init__(self, in_channel):
        super(Encoder, self).__init__()
        self.in_channel = in_channel
        self.e1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channel),
            nn.Conv2d(in_channels=self.in_channel, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.e2 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

        )
        self.e3 = nn.Sequential(
            #nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

        )

    def forward(self, x):
        y1 = self.e1(x)
        y2 = self.e2(y1)
        y3 = self.e3(y2)
        return y1,y2,y3
