import os
import torchio as tio
import SimpleITK as sitk
from torch.utils.data import DataLoader

from core.config import config


def get_trainloader(root):
    names = os.listdir(root)

    subjects = []
    for name in names:
        subject = tio.Subject(
            data = tio.ScalarImage(os.path.join(root, name))
        )
        subjects.append(subject)
    transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.CropOrPad(target_shape=[182]+config.TRAIN.PATCH_SIZE)
    ])
    dataset = tio.SubjectsDataset(subjects, transforms)
    queue = tio.Queue(
        dataset,
        config.TRAIN.QUEUE_LENGTH,
        config.TRAIN.SAMPLES_PER_VOLUME,    
        tio.UniformSampler([1]+config.TRAIN.PATCH_SIZE),
        config.NUM_WORKERS
    )
    loader = DataLoader(queue, config.TRAIN.BATCH_SIZE, num_workers=0)
    return loader
