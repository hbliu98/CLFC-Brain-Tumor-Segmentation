import os
import time
import torch
import random
import logging
import numpy as np


def create_logger(out_dir, logfile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logfile = os.path.join(out_dir, logfile[:-4]+'@'+time.strftime('(%m-%d)-%H-%M-%S')+'.log')

    handler = logging.FileHandler(logfile, mode='w')
    handler.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, times=1):
        self.val = val
        self.sum += val * times
        self.count += times
        self.avg = self.sum / self.count


def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))
