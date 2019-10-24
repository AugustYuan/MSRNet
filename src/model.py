from MsrNet_Module import MsrNet
import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataset import Dataset
from loss import  loss_fn
import cv2


class MSRMODEL(nn.Module):
    def __init__(self, config):
        super(MSRMODEL, self).__init__()
        self.num_class = config.num_class
        self.lr = config.LR
        self.model = MsrNet(config.IN_CHANNEL, self.num_class).cuda()
        self.weight = torch.from_numpy(config.WEIGHT).cuda().float()
        self.iteration = int(float(config.iteration))
        self.batch_size = config.BATCH_SIZE
        self.load_path = config.LOAD_PATH
        self.save_path = config.SAVE_PATH
        #self.add_module("model", model)

        if len(config.DEVICE_ID) > 1:
            self.model = nn.DataParallel(self.model, device_ids=config.DEVICE_ID)

        if config.IS_LOAD:
            self.load()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.STEP_SIZE,
            gamma=0.8)

        self.train_dataset = Dataset(config.t_img_path, config.t_label_path, config.t_gse_path)
        self.val_dataset = Dataset(config.v_img_path, config.v_label_path, config.v_gse_path)

    def load(self):
        if self.load_path is not None:
            self.model.load_state_dict(torch.load(self.load_path))

    def save(self, epoch):
        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path+str(epoch)+'.pkl')
        else:
            torch.save(self.model.state_dict(), './' + str(epoch) + '.pkl')

    def to_cuda(self, *args):
        return (item.cuda() for item in args)

    def IOU(self, pre, lable, num_class):
        pre = np.int64(pre.reshape(-1))
        lable = np.int64(lable.reshape(-1))
        hist = np.bincount(num_class * lable + pre, minlength=num_class ** 2).reshape(num_class, num_class)
        IOU = np.diag(hist) / (np.sum(hist, axis=0) + np.sum(hist, axis=1) - np.diag(hist))
        mIOU = np.nanmean(IOU)
        return mIOU * 100

    def train(self, mode=True):

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            drop_last=True,
            shuffle=False
        )

        if len(self.train_dataset) == 0:
            print("No training data was provided!")
            return
        generation = 0
        for epoch in range(self.iteration):

            print('\n\nTraining epoch: %d' % epoch)

            self.scheduler.step()
            print("Learning_rate: ", self.scheduler.get_lr()[0])
            output3 = 0
            label3 = 0
            gse3 = 0
            loss = 0

            for items in train_loader:

                generation += 1

                img, label3, label2, label1, gse3, gse2 = self.to_cuda(*items)

                self.optimizer.zero_grad()

                output1, output2, output3 = self.model(img)
                loss = loss_fn(output1, output2, output3,
                               label1.long(), label2.long(), label3.long(),
                               gse2, gse3, self.weight)

                loss.backward()
                self.optimizer.step()

            output3 = np.argmax(output3.data.cpu().numpy(), 1)
            label3 = np.squeeze(label3.data.cpu().numpy())
            gse3 = gse3.data.cpu().numpy()
            miou = self.IOU(output3, label3, self.num_class)
            acc = np.equal(output3, label3)
            gse_acc = np.sum(acc*gse3) / np.sum(gse3)
            acc = np.mean(acc)
            print('----------Loss = {:5.5f} ;  Accuracy = {:2.2f}% ---------'.format(np.float64(loss), acc*100))
            print('----------GSE_acc = {:2.2f}%;   mIOU = {:2.2f}% ---------'.format(gse_acc, miou))

            cv2.imwrite('../pic3/{}.png'.format(epoch), output3[0].reshape(512, 512) * 20)

            if epoch % 10 == 0:
                self.save(epoch)