import os
import pickle
import numpy as np
import torchio as tio

from core.config import config


def get_validset(fold):
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    valids = splits['val']

    subjects = []
    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name+'.npz'))['data']
        subject = tio.Subject(
            data = tio.ScalarImage(tensor=data[:-1]),
            label = tio.LabelMap(tensor=data[-1:]),
            name = name
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    return dataset
