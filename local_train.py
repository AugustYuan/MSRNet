import os
import torch
import logging
import warnings
import argparse
from model_train import Model
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def set_():
    warnings.filterwarnings('ignore')
    logging.basicConfig(format='%(asctime)s -- %(levelname)s：  %(message)s', level=logging.INFO)

def get_args():
    parser = argparse.ArgumentParser(description='Train MsrNet')
    parser.add_argument('config', help='train config file path')
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()

def main():
    set_()
    args = get_args()
    torch.cuda.set_device(2)
    #torch.distributed.init_process_group(backend='nccl', init_method='env://')
    assert os.path.isfile(args.config), ValueError('{} is not a config file.'.format(args.config))
    logging.info('Building model from {} ...'.format(args.config))
    #print('local rank:', args.local_rank)
    model = Model(args.config)
    model.model = torch.nn.DataParallel(model.model,device_ids=[2,3])
    #model.model = torch.nn.parallel.DistributedDataParallel(model.model,
    #                                              device_ids=[args.local_rank],
    #                                              output_device=args.local_rank)
    #sampler = DistributedSampler(model.dataset_train)
    #model.dataloader_train = DataLoader(model.dataset_train, sampler=sampler)
    model.train()


if __name__ == "__main__":
    main()
