from torch.utils.data import DataLoader
import glob
import cv2
import torch
import numpy as np
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_path, lab_path, gse_path):
        super(Dataset, self).__init__()
        self.data = list(glob.glob(img_path+"/*.bmp"))
        self.label = list(glob.glob(lab_path+"/*.bmp"))
        self.gse = list(glob.glob(gse_path+"/*.bmp"))
        self.data.sort()
        self.label.sort()
        self.gse.sort()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print("Loading error: "+self.data[index])
            item = self.load_item(1)
        return item

    def to_tensor(self, img):
        img = torch.from_numpy(img.transpose(2,0,1))
        return img

    def load_item(self, index):
        lenth = 512
        img = cv2.imread(self.data[index],0).reshape(lenth,lenth,1)
        label3 = cv2.imread(self.label[index], 0).reshape(lenth,lenth,1)
        label2 = cv2.resize(label3, (lenth//2,lenth//2), cv2.INTER_NEAREST).reshape(lenth//2,lenth//2,1)
        label1 = cv2.resize(label3, (lenth//4,lenth//4), cv2.INTER_NEAREST).reshape(lenth//4,lenth//4,1)
        gse3 = cv2.imread(self.gse[index], 0).reshape(lenth,lenth,1)
        gse2 = cv2.resize(gse3, (lenth//2,lenth//2), cv2.INTER_NEAREST).reshape(lenth//2,lenth//2,1)
        return self.to_tensor(img).float(),\
               torch.squeeze(self.to_tensor(label3)).long(),\
               torch.squeeze(self.to_tensor(label2)).long(),\
               torch.squeeze(self.to_tensor(label1)),\
               self.to_tensor(gse3).float(),\
               self.to_tensor(gse2).float()



