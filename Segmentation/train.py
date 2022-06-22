import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models.network import SegNet, prediction, projection
from core.config import config
from core.scheduler import PolyScheduler
from core.function import train, inference
from core.loss import DiceCELoss
from dataset.dataset import get_validset
from dataset.dataloader import get_trainloader
from dataset.augmenter import get_train_generator
from utils.utils import determine_device, save_checkpoint, create_logger, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', help='which data fold to train on', required=True, type=int)
    args = parser.parse_args()
    return args


def main(args):
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = nn.ModuleDict({
        'segnet': SegNet(),
        'projections': nn.ModuleList([
            projection(nf) for nf in [32, 64, 128, 256]
        ]),
        'predictions': nn.ModuleList([
            prediction() for _ in range(4)
        ])
    })
    if config.TRAIN.PARALLEL:   # only cuda is supported
        devices = config.TRAIN.DEVICES
        model = nn.DataParallel(model, devices).cuda(devices[0])
    else:   # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.95, nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)
    # deep supervision weights, normalize sum to 1
    criterion = DiceCELoss()
    # training data generator
    trainloader = get_trainloader(args.fold)
    train_generator = get_train_generator(trainloader, scales=(1., 0.5, 0.25, 0.125))
    # validation dataset
    validset = get_validset(args.fold)

    best_model = False
    best_perf = 0.0
    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate : {}'.format(optimizer.param_groups[0]['lr']))

        train(model, train_generator, optimizer, criterion, logger, config, epoch)
        scheduler.step()
        # running validation at every epoch is time consuming
        if epoch%config.VALIDATION_INTERVAL == 0:
            if config.TRAIN.PARALLEL:
                perf = inference(model.module['segnet'], validset, logger, config)
            else:
                perf = inference(model['segnet'], validset, logger, config)

        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf,
            'optimizer': optimizer.state_dict(),
        }, best_model, config.OUTPUT_DIR, filename='checkpoint.pth')


if __name__ == '__main__':
    args = parse_args()
    main(args)
