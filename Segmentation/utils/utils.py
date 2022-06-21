import os
import time
import torch
import random
import logging
import numpy as np
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d


def determine_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def update_config(config, cfgfile):
    config.defrost()
    config.merge_from_file(cfgfile)
    config.freeze()


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


def get_patch_size(final_patch_size, rot_x, rot_y, rot_z, scale_range):
    """
    to avoid loss of information during data augmentation, we need to input larger patch to the augmenter,
    the larger patch size can be computed according to rotation and scale range
    """
    if isinstance(rot_x, (tuple, list)):
        rot_x = max(np.abs(rot_x))
    if isinstance(rot_y, (tuple, list)):
        rot_y = max(np.abs(rot_y))
    if isinstance(rot_z, (tuple, list)):
        rot_z = max(np.abs(rot_z))
    rot_x = min(90 / 360 * 2. * np.pi, rot_x)
    rot_y = min(90 / 360 * 2. * np.pi, rot_y)
    rot_z = min(90 / 360 * 2. * np.pi, rot_z)
    coords = np.array(final_patch_size)
    final_shape = np.copy(coords)
    if len(coords) == 3:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
        final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
    elif len(coords) == 2:
        final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
    final_shape /= min(scale_range)
    return final_shape.astype(int)
