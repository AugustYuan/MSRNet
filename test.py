import torch
from src.loss import Batch_Balance_CE

num_class = 10
a = torch.zeros(20,num_class)
b = torch.tensor(list(range(10))*2).long()


def forward(logits, label):
    #assert logits.shape[-2:] == label.shape[-2:], ValueError('Lossing match between logit and label.')
    #assert logits.shape[1] == num_class, KeyError('Error category.')
    loss_ce = torch.nn.CrossEntropyLoss(reduce=False)(logits, label)
    label = label.reshape(-1)
    loss_ce = loss_ce.reshape(-1)
    one_hot = torch.zeros(label.shape[0], num_class)
    one_hot[range(label.shape[0]), label] = 1.0
    f_weight = torch.sum(one_hot, 0) / torch.sum(one_hot)
    f_weight = torch.exp(-num_class * f_weight)
    ce_data = loss_ce.data.reshape(-1, 1)
    l_weight = ce_data * (one_hot.float())
    l_weight = torch.tanh(torch.mean(ce_data * (one_hot.float()), 0))
    one_hot = torch.sum((f_weight + l_weight).reshape(1, -1) * one_hot, -1)
    loss = torch.mean(loss_ce * one_hot)
    return loss

k = forward(a, b)

print(k)
