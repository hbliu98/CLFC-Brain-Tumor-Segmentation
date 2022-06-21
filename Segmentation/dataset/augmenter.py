import numpy as np
from core.config import config
from skimage.transform import resize
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class DownsampleSegTransform(AbstractTransform):
    """
    transform segmentation label to a list of labels scaled according to deep supervision scales using nearestNeighbor interpolation
    """
    def __init__(self, scales=(1., 0.5, 0.25), label_key='label'):
        super().__init__()
        self.scales = scales
        self.label_key = label_key

    def __call__(self, **data_dict):
        label = data_dict[self.label_key]
        axes = list(range(2, len(label.shape)))
        labels = []
        for s in self.scales:
            if s == 1.:
                labels.append(label > 0)    # only wt segmentation is supported
            else:
                new_shape = np.array(label.shape).astype(float)
                for a in axes:
                    new_shape[a] *= s
                new_shape = np.round(new_shape).astype(int)
                out_label = np.zeros(new_shape, dtype=label.dtype)
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        out_label[i, j] = resize(label[i, j].astype(float), new_shape[2:], order=0, mode='edge', clip=True, anti_aliasing=False).astype(label.dtype)
                labels.append(out_label > 0)    # only wt segmentation is supported
        data_dict[self.label_key] = labels
        return data_dict


def get_train_generator(trainloader, scales):
    angle_x = (-np.pi, np.pi)
    angle_y = (0., 0.)
    angle_z = (0., 0.)
    # for mirror
    mirror_axes = (0, 1)
    
    transforms = []
    # spatial transformation 
    transforms.extend([
        SpatialTransform(
            # output patch size
            patch_size=config.TRAIN.PATCH_SIZE,
            # how to get data and label
            data_key='data', label_key='label',
            # rotation
            do_rotation=True, angle_x=angle_x, angle_y=angle_y, angle_z=angle_z, p_rot_per_sample=0.2,
            # scaling
            do_scale=True, scale=(0.7, 1.4), p_scale_per_sample=0.2,
            # others
            border_mode_data='constant',
            do_elastic_deform=False, random_crop=False
        )
    ])
    # Gaussian Noise, Blur
    transforms.extend([
        GaussianNoiseTransform(p_per_sample=0.1),
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.),
            different_sigma_per_channel=True,
            p_per_sample=0.2,
            p_per_channel=0.5
        )
    ])
    # Brightness, Contrast
    transforms.extend([
        BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15),
        ContrastAugmentationTransform(p_per_sample=0.15)
    ])
    # Low resolution
    # interpolation order in skimage: 0：Nearest-neighbor 1：Bi-linear(默认) 2：Bi-quadratic 3：Bi-cubic 4：Bi-quartic 5：Bi-quintic
    # per_channel = True can simulate low resolution of random modalities <- here channel equals modality
    transforms.extend([
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1.),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25
        )
    ])
    # Invert Gamma, Gamma
    transforms.extend([
        GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=True,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.1
        ),
        GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=False,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.3
        )
    ])
    # Mirror
    transforms.extend([MirrorTransform(axes=mirror_axes, data_key='data', label_key='label')])
    # for deep supervision, yooo, ignore lowest two resolution
    transforms.extend([DownsampleSegTransform(scales=scales, label_key='label')])
    # ToTensor
    transforms.extend([NumpyToTensor(keys=['data', 'label'], cast_to='float')])
    
    transforms = Compose(transforms)

    batch_generator = MultiThreadedAugmenter(
        data_loader=trainloader,
        transform=transforms,
        num_processes=config.NUM_WORKERS,
        pin_memory=True
    )

    return batch_generator
