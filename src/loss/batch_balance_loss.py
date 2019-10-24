import torch
import torch.nn as nn


class Batch_Balance_CE():
    def __init__(self, num_class):
        self.num_class = num_class
        self.CE = nn.CrossEntropyLoss(reduce=False)

    def forward(self, logits, label):
        assert logits.shape[-2:] == label.shape[-2:], ValueError('Lossing match between logit and label.')
        assert logits.shape[1] == self.num_class, KeyError('Error category.')
        loss_ce = self._batch_CE(logits, label)
        label = label.reshape(-1)
        loss_ce = loss_ce.reshape(-1)
        one_hot = torch.zeros(label.shape[0], self.num_class).cuda()
        one_hot[range(label.shape[0]), label] = 1.0
        f_weight = torch.sum(one_hot, 0)/torch.sum(one_hot)
        f_weight = torch.exp(-self.num_class*f_weight)
        ce_data = loss_ce.data.reshape(-1,1)
        l_weight = torch.tanh(torch.mean(ce_data*(one_hot.float()), 0))
        one_hot = torch.sum((f_weight+l_weight).reshape(1,-1) * one_hot, -1)
        loss = torch.mean(loss_ce*one_hot)
        return loss

    def _batch_CE(self, logits, label):
        return self.CE(logits, label)