import os
import pickle
import numpy as np
from core.config import config
from collections import OrderedDict
from utils.utils import get_patch_size
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase


class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1/3
        # larger patch size is required for proper data augmentation
        self.patch_size = get_patch_size(patch_size, (-np.pi, np.pi), (0, 0), (0, 0), (0.7, 1.4))
    
    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form slice
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])['data']
            # slice with foreground class presented should be selected ?
            if i < round(self.batch_size * (1 - self.oversample_foreground_percent)):
                force_fg = False
            else:
                force_fg = True
            if force_fg:
                # select slice containing foreground class
                locs = self._data[name]['locs']
                cls = np.random.choice(list(locs.keys()))
                indices = locs[cls][:, 0]   # all axial indices
                sel_idx = np.random.choice(np.unique(indices))
                data = data[:, sel_idx]
                # pad slice centered at selected location
                # the idea is simple: if selected location is biased towards left, then left side will be padded more pixels. note that we cannot crop slice, so the minimum pad length is zero
                loc = locs[cls][indices == sel_idx]
                loc = loc[np.random.choice(len(loc))][1:]
                shape = np.array(data.shape[1:])
                center = shape // 2
                bias = loc - center
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2 - bias
                pad_right = pad_length - pad_length // 2 + bias
                pad_left = np.clip(pad_left, 0, pad_length)
                pad_right = np.clip(pad_right, 0, pad_length)
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            else:
                # randomly select slice
                sel_idx = np.random.choice(data.shape[1])
                data = data[:, sel_idx]
                shape = np.array(data.shape[1:])
                pad_length = self.patch_size - shape
                pad_left = pad_length // 2
                pad_right = pad_length - pad_length // 2
                data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}


def get_trainloader(fold):
    # list data path and properties
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    trains = splits['train']
    dataset = OrderedDict()
    for name in trains:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name+'.npz')
        with open(os.path.join(config.DATASET.ROOT, name+'.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)
    
    return DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)
