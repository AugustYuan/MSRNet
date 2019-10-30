import torch
import numpy as np


def iou(pre, lable, num_class):
    pre = np.int64(pre.reshape(-1))
    lable = np.int64(lable.reshape(-1))
    assert pre.shape[0] == lable.shape[0]
    hist = np.bincount(num_class*lable+pre, minlength=num_class**2).reshape(num_class, num_class)
    IOU = np.diag(hist)/(np.sum(hist,axis=0)+np.sum(hist, axis=1)-np.diag(hist))
    mIOU = np.nanmean(IOU)
    acc = np.nanmean(np.diag(hist)/np.sum(hist, axis=1))
    return mIOU*100, acc*100

def iou_cuda(pre, lable, num_class):
    pre = pre.int().view(-1)
    lable = lable.int().view(-1)
    assert pre.size() == lable.size()
    hist = torch.zeros(num_class**2)
    mask = (lable<num_class) & (lable>=0)
    pre = pre[mask]
    lable = lable[mask]
    hist = ((lable*num_class+pre).long()).bincount(minlength=num_class**2)
    hist = hist.view(num_class, num_class).float()
    ious = hist.diag()/(torch.sum(hist, 0)+torch.sum(hist, 1)-hist.diag())
    mask = ious<=1.0
    miou = torch.mean(ious[mask])
    accs = torch.sum(hist.diag())/torch.sum(hist)
    mask = accs<1.0
    accs = torch.mean(accs[mask])
    return miou*100.0, accs*100.0
