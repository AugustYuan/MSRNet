import os
import time
import torch
import logging
import traceback
import matplotlib
import numpy as np
import torch.nn as nn
from models import MsrNet
from dataset import MsrDataset
from collections import OrderedDict
from src.loss import Batch_Balance_CE
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import load_config, iou_cuda, iou


class Model(nn.Module):
    def __init__(self, cfg, rank=0):
        super(Model, self).__init__()
        self.cfg = cfg
        if isinstance(self.cfg, str):
            self.cfg = load_config(self.cfg)
        elif isinstance(self.cfg, dict):
            pass
        else:
            raise Exception('Error cfg input')
        self._init_logger()
        self.rank=rank
        self.model_cfg = self.cfg['model']
        self.data_cfg = self.cfg['data']
        self.work_dir = self.cfg['work_dir']
        self.cuda_flag = False
        self._build_model()
        self._init_weights(self.modules())
        if self.cuda_flag:
            self.model.cuda()
        # init dataset
        self.dataset_train, self.dataset_eval = None, None
        if 'train' in self.data_cfg.keys():
            self.dataset_train = self._build_dataset(mode='train')
            self.dataloader_train = self._build_dataLoader(self.dataset_train, self.cfg['train_cfg']['batch_size'])
        if 'eval' in self.data_cfg.keys():
            self.dataset_eval =self._build_dataset(mode='eval')
        assert self.dataset_eval or self.dataset_train, ValueError('No dataset was provided.')

        if self.cfg.get('work_dir') and os.path.isdir(self.cfg['work_dir']):
            self.work_dir = self.cfg['work_dir']
            if self.work_dir.endswith('/'):
                self.work_dir = self.work_dir[:-1]
        else:
            self.work_dir = os.path.join(self.cfg['root'], 'trained_model')
            os.mkdir(self.work_dir)

        self.loss_fn = self._build_loss()
        self.loss_weight = torch.tensor(self.cfg['loss_weight'])
        self._build_optimizer()
        self.print_freq = self.cfg['print_freq']

    def _build_model(self):
        self.model = MsrNet(self.cfg['model'])
        if self.cfg['with_gpu'] and torch.cuda.is_available():
            self.cuda_flag=True
            #self.model.cuda()

    def _build_dataLoader(self,
                          dataset,
                          batch_size,
                          drop_last=True,
                          shuffle=True):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle
        )
        return dataloader

    def _build_dataset(self, mode='train'):
        cfg = self.data_cfg[mode]
        logging.info('Building {} dataset from {} ...'.format(mode, os.path.join(self.cfg['root'], cfg['img_path'])))
        dataset = MsrDataset(
            ImgPath=os.path.join(self.cfg['root'], cfg['img_path']),
            LabelPath=os.path.join(self.cfg['root'], cfg['label_path']),
            img_size=cfg['img_size'],
            imgSty=cfg['imgSty'],
            labSty=cfg['labSty'],
            is_train=cfg['is_train'],
            is_augment=cfg['is_augment'],
            flip_rate=cfg['flip_rate'],
            gause_rate=cfg['gause_rate'],
            salt_rate=cfg['salt_rate']
        )
        return dataset

    def _init_weights(self, model):
        for m in model:
            if isinstance(m, nn.Sequential):
                self._init_weights(m)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 1.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _save_checkpoint(self, epoch, iter):
        check_point = self.model.state_dict()
        result = dict(
            state_dict=check_point,
            meta=dict(
                epoch=epoch,
                iteration=iter
            )
        )
        path = os.path.join(self.work_dir, 'epoch_{}.pkl'.format(epoch))
        logging.info('Save checkpoints to {} ...'.format(path))
        torch.save(result, path)

    def _resume_model(self):
        logging.info('Resuming from {} ...'.format(self.cfg['resume']['resume_dir']))
        self._load_checkpoints(self.cfg['resume']['resume_dir'])

    def _load_checkpoints(self, dir):
        if os.path.isfile(dir):
            try:
                checkpoint = torch.load(dir, map_location='cpu')
                if isinstance(checkpoint, OrderedDict):
                    state_dict = checkpoint
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                shape_mismatch_pairs = []
                unexpected_keys = []
                model_state = self.model.state_dict()
                for name, param in state_dict.items():
                    if name not in model_state:
                        unexpected_keys.append(name)
                        continue
                    if isinstance(param, torch.nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    if param.size() != model_state[name].size():
                        shape_mismatch_pairs.append(
                            [name, model_state[name].size(),
                             param.size()])
                        continue
                    model_state[name].copy_(param)
                all_missing_keys = set(model_state.keys()) - set(state_dict.keys())
                missing_keys = [key for key in all_missing_keys]
                log = dict(
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys
                )
                logging.error(log)
            except:
                print('Loading fail.')

    def train(self, mode=True):
        assert self.dataset_train is not None, ValueError('No train data is provided.')
        #dataloader = self._build_dataLoader(self.dataset_train, self.cfg['train_cfg']['batch_size'])
        length = len(self.dataloader_train)
        max_epoch = self.cfg['train_cfg']['max_epochs']
        max_iter = self.cfg['train_cfg']['max_epochs']
        logging.info('Start training with Max epoch: {}'.format(max_epoch))
        epoch = 0
        iteration = 0
        while epoch <= max_epoch:
            epoch += 1
            self.scheduler.step()
            lr_ = self.scheduler.get_lr()[0]
            for i, data_batch in enumerate(self.dataloader_train):
                with torch.enable_grad():
                    try:
                        iteration += 1
                        self.optimizer.zero_grad()
                        start = time.time()
                        output = self._batch_prosess(*data_batch)
                        output['loss'].backward()
                        self.optimizer.step()
                        end = time.time()
                        if iteration % self.print_freq == 0 and self.rank == 0:
                            _, logit = torch.max(output['model_output'][-1], 1)
                            label = data_batch[1][-1]
                            miou, acc = self._iou(logit, label)
                            logging.info(
                                'Epoch: [{}] [{}/{}], Lr: {:.5f}, Time: {:.2f}, Loss: {:.4f}, mIOU: {:.2f}, Acc: {:.2f}'.format(
                                    epoch, i, length, lr_, end-start, output['loss'], miou.data, acc.data
                                )
                            )

                    except Exception as e:
                        print(str(e) + '\n' + traceback.format_exc())
            if self.rank == 0:
                self._save_checkpoint(epoch, iteration)

    def _build_loss(self):
        loss_type = {
            'BB_CE': Batch_Balance_CE(self.cfg['num_class']).forward,
            "CE": nn.CrossEntropyLoss()
        }
        loss_fn = loss_type[self.cfg['loss_type']]
        return loss_fn

    def _build_optimizer(self):
        optim_list = {
            'SGD':torch.optim.SGD
        }
        self.optimizer = optim_list[self.cfg['train_cfg']['optimizer']](
            self.model.parameters(),
            lr=self.cfg['train_cfg']['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.cfg['train_cfg']['step_size'],
            gamma=self.cfg['train_cfg']['gamma'])

    def _batch_prosess(self, img, labels):
        model_output = self.model(img)
        assert len(model_output) == len(labels), \
            ValueError("length of model_output {} miss match with length of labels {}".format(
                len(model_output), len(labels)))
        loss = []
        for logit, label in zip(model_output, labels):
            loss.append(
                self.loss_fn(logit, label)
            )
        loss = sum(l*w for l,w in zip(loss,self.loss_weight))
        output = dict(
            model_output=model_output,
            loss=loss
        )
        return output

    def _iou(self, logit, lable):
        if self.cfg['with_gpu'] and torch.cuda.is_available():
            return iou_cuda(logit, lable, self.cfg['num_class'])
        else:
            return iou(logit, lable, self.cfg['num_class'])

    def _init_logger(self):
        logging.basicConfig(format='%(asctime)s -- %(levelname)s：  %(message)s', level=logging.INFO)


