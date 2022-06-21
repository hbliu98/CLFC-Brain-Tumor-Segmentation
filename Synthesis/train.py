import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import *
from core.config import config
from core.function import train
from models.network import Encoder, Decoder
from dataset.dataset import get_trainloader
from utils.utils import save_checkpoint, create_logger, setup_seed


def main():
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    enc = Encoder(cdim=1).to(torch.device(config.DEVICE))
    dec = Decoder(cdim=1).to(torch.device(config.DEVICE))
    optimE = optim.SGD(enc.parameters(), lr=1e-4)
    optimD = optim.SGD(dec.parameters(), lr=5e-3)
    
    trainloader = get_trainloader(config.DATASET.ROOT)

    logger = create_logger('log', 'train.log')
    for epoch in range(config.TRAIN.EPOCH):
        
        train([enc, dec], trainloader, optimE, optimD, logger, config, epoch)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'enc': enc.state_dict(),
            'dec': dec.state_dict(),
            'optimE': optimE.state_dict(),
            'optimD': optimD.state_dict(),
        }, False, config.OUTPUT_DIR, filename='checkpoint.pth')


if __name__ == '__main__':
    main()
