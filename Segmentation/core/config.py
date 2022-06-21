from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 6
config.PRINT_FREQ = 10
config.VALIDATION_INTERVAL = 5
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = 'DATA/preprocessed/brats19'

config.TRAIN = CN()
config.TRAIN.LR = 1e-2
config.TRAIN.WEIGHT_DECAY = 3e-5
config.TRAIN.BATCH_SIZE = 4
config.TRAIN.PATCH_SIZE = [192, 160]
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.EPOCH = 200
config.TRAIN.PARALLEL = False
config.TRAIN.DEVICES = [0]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 4
config.INFERENCE.PATCH_SIZE = [192, 160]
config.INFERENCE.PATCH_OVERLAP = [96, 80]
