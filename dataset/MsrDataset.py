import os
import cv2
import glob
import torch
import logging
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision.transforms import ToTensor


class MsrDataset(Dataset):
    def __init__(self,
                 ImgPath,
                 LabelPath,
                 img_size=512,
                 imgSty='.jpg',
                 labSty='.png',
                 is_train=True,
                 is_augment=False,
                 flip_rate=0.5,
                 gause_rate=0.5,
                 salt_rate=0.5,
                 cuda=True):
        super(MsrDataset, self).__init__()
        assert os.path.isdir(ImgPath), KeyError('{} is not an existing dir.'.format(ImgPath))
        assert os.path.isdir(LabelPath), KeyError('{} is not an existing dir.'.format(LabelPath))
        self.size = img_size
        self.is_train = is_train
        self.cuda = cuda
        self.mode = 'Train/Eval' if is_train else 'Test'
        logging.info('Setting dataset ...')
        logging.info('Dataset Mode: {}'.format(self.mode))
        self.imgdir = os.path.join(ImgPath, '*'+imgSty)
        self.labdir = os.path.join(LabelPath, '*'+labSty)
        self.img_files = list(glob.glob(self.imgdir))
        self.lab_files = list(glob.glob(self.labdir))
        assert len(self.img_files)>0 and len(self.lab_files)>0, ValueError('No data was provided.')
        assert len(self.img_files) == len(self.lab_files), ValueError('Losing match between images and labels.')
        self.lab_files.sort()
        self.img_files.sort()
        self.is_augment = is_augment
        if self.is_augment:
            self.flip_rate = flip_rate
            assert flip_rate<=1.0 and flip_rate>=0.0,\
                ValueError('The flip rate should between 0.0 and 1.0, but get {}'.format(flip_rate))
            self.gause_rate = gause_rate
            assert gause_rate <= 1.0 and gause_rate >= 0.0, \
                ValueError('The gause_rate should between 0.0 and 1.0, but get {}'.format(gause_rate))
            self.salt_rate = salt_rate
            assert salt_rate <= 1.0 and salt_rate >= 0.0, \
                ValueError('The salt_rate should between 0.0 and 1.0, but get {}'.format(salt_rate))


    def __len__(self):
        return len(self.img_files)
    @property
    def path(self):
        return {
            'Img_Path':self.imgdir[:-5],
            'Lab_Path':self.labdir[:-5]
        }

    def __getitem__(self, index):
        try:
            item = self._load_item(index)
        except:
            logging.info('Loading Error: img_file:{}; lab_file:{}.'\
                          .format(self.img_files[index], self.lab_files[index]))
            item = self._load_item(1)
        return item

    def _load_item(self, index):
        img = cv2.imread(self.img_files[index], 0)
        img = cv2.resize(img, (self.size, self.size), cv2.INTER_CUBIC)
        if not self.is_train:
            return (self._to_tensor(img))
        label3 = cv2.imread(self.lab_files[index], 0)
        label3 = cv2.resize(label3, (self.size, self.size), cv2.INTER_NEAREST)
        if self.is_augment:
            img, label3 = self._augument(img, label3)
        label2 = cv2.resize(label3, (self.size//2, self.size//2), cv2.INTER_NEAREST)
        label1 = cv2.resize(label3, (self.size//4, self.size//4), cv2.INTER_NEAREST)
        return(self._to_tensor(img),
               [self._to_tensor(label1, True),
               self._to_tensor(label2, True),
               self._to_tensor(label3, True)])

    def _augument(self, img, label):
        assert isinstance(img, np.ndarray), ValueError('Error input type.')
        assert isinstance(label, np.ndarray), ValueError('Error input type.')
        flag = torch.rand(1) > 1.0-self.flip_rate
        if flag:
            img = cv2.flip(img,1)
            label = cv2.flip(label,1)
        flag = torch.rand(1) > 1.0-self.gause_rate
        if flag:
            img = self._gaususs_noise(img)
        flag = torch.rand(1) > 1.0-self.salt_rate
        if flag:
            img = self._salt_noise(img)
        return img,label

    def _gaususs_noise(self, img, mean=0.0, var=.0001):
        img = np.float(img)/255
        noise = np.random.normal(mean, var**0.5, img.shape)
        img += noise
        img -= np.min(img)
        img /= np.max(img)
        return np.uint8(img*255)

    def _salt_noise(self, img, thre=0.01):
        mask = np.random.rand(img.shape)
        img[mask<thre] = 0
        img[mask>(1.0-thre)] = 255
        return img



    def _to_tensor(self, arr, is_label=False):
        assert isinstance(arr, np.ndarray), ValueError('Error input type.')
        if self.cuda:
            if is_label:
                return torch.squeeze(torch.from_numpy(arr)).long().cuda()
            else:
                return (ToTensor()(arr)*255.).cuda()
        else:
            if is_label:
                return torch.squeeze(torch.from_numpy(arr).long())
            else:
                return ToTensor()(arr)*255
