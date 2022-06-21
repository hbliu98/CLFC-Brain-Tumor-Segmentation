import torch
import argparse
import torch.nn as nn

from models.network import SegNet
from core.config import config
from core.function import inference
from dataset.dataset import get_validset
from utils.utils import determine_device, create_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', help='path for pretrained weights', required=True, type=str)
    parser.add_argument('--fold', help='which fold to validate', required=True, type=int)
    args = parser.parse_args()
    return args


def main(args):
    if config.TRAIN.PARALLEL:   # only cuda is supported
        devices = config.TRAIN.DEVICES
        model = SegNet()
        model = nn.DataParallel(model, devices).cuda(devices[0])
    else:   # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = SegNet().to(device)
    # load pretrained weights
    model.load_state_dict(torch.load(args.weights))
    # validation dataset
    validset = get_validset(args.fold)

    logger = create_logger('log', 'test.log')
    inference(model, validset, logger, config)


if __name__ == '__main__':
    args = parse_args()
    main(args)
