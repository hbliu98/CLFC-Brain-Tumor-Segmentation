from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 6
config.PRINT_FREQ = 10
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345
config.DEVICE = 'cuda'

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = 'DATA/ixi'

config.TRAIN = CN()
config.TRAIN.BATCH_SIZE = 120
config.TRAIN.PATCH_SIZE = [256, 256]
config.TRAIN.QUEUE_LENGTH = 3840
config.TRAIN.SAMPLES_PER_VOLUME = 32
config.TRAIN.EPOCH = 200
