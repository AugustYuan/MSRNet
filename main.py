from configs.Config import config
from model import MSRMODEL
import numpy as np
import torch
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(50)
default_config = config()
model = MSRMODEL(default_config)

if __name__ == '__main__':
    model.train()