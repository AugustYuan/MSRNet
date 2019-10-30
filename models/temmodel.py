import torch.nn as nn
from .encoder import Encoder
from .repair_net import RepairNets, Repairnet
from .ssdecoder import SSDecoder
from collections import defaultdict



class temModel(nn.Module):
    def __init__(self, cfg):
        super(temModel, self).__init__()
        assert isinstance(cfg, dict)
        self.cfg = cfg
        self.sub_model_list = list(cfg.keys())
        self.m_list = {
            'encoder': Encoder,
            'repairnet': RepairNets,
            'ssdecoder': SSDecoder
        }
        self.encoder = Encoder(cfg['encoder'])
        self.decoder = SSDecoder(cfg['ssdecoder'])
        self.repair1 = Repairnet(128, 11, ['RCU','IRU'])
        self.repair2 = Repairnet(64, 11, ['RCU', 'IRU'])

    def forward(self, img):
        feat = self.encoder(img)
        output = []
        res1 = self.decoder(feat[0])
        output.append(res1)
        res2 = self.repair1(feat[1], res1)
        output.append(res2)
        res3 = self.repair2(feat[2], res2)
        output.append(res3)
        return output