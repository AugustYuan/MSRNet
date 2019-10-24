import torch.nn as nn
from .encoder import Encoder
from .repair_net import RepairNets, Repairnet
from .ssdecoder import SSDecoder
from collections import defaultdict


class MsrNet(nn.Module):
    def __init__(self, config):
        super(MsrNet, self).__init__()
        assert isinstance(config, dict)
        self.cfg = config
        self.sub_model_list = list(config.keys())
        self.m_list = {
            'encoder':Encoder,
            'repairnet':RepairNets,
            'ssdecoder':SSDecoder
        }
        self.encoder, self.ssdecoder, [self.repair1, self.repair2] = self._build_model()

    def forward(self, img):
        output=[]
        feats = self.encoder(img)
        res1 = self.ssdecoder(feats[0])
        output.append(res1)
        res2 = self.repair1(feats[1], res1)
        output.append(res2)
        res3 = self.repair2(feats[2], res2)
        output.append(res3)
        return output


    def _build_model(self):
        self.sub_model_list.sort()
        assert len(self.sub_model_list) == 3
        assert list(self.m_list.keys())[0] in self.sub_model_list
        assert list(self.m_list.keys())[1] in self.sub_model_list
        assert list(self.m_list.keys())[2] in self.sub_model_list
        model = defaultdict()
        for m in self.sub_model_list:
            assert m in self.m_list, KeyError("Model has no contribute to {}.".format(m))
            model[m] = self.m_list[m](self.cfg[m])
        return model['encoder'], model['ssdecoder'], model['repairnet']