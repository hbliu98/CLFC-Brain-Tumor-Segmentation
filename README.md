# CLFC Brain Tumor Segmentation
> This is the code for MICCAI 2022 paper: "Multimodal Brain Tumor Segmentation Using Contrastive Learning based Feature Comparison with Monomodal Normal Brain Images". The training pipeline is motivated by [nnUNet](https://github.com/MIC-DKFZ/nnUNet)

If you find it useful in your own research, please cite our paper:
```
@inproceedings{liu2022multimodal,
  title={Multimodal Brain Tumor Segmentation Using Contrastive Learning Based Feature Comparison with Monomodal Normal Brain Images},
  author={Liu, Huabing and Nie, Dong and Shen, Dinggang and Wang, Jinda and Tang, Zhenyu},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={118--127},
  year={2022},
  organization={Springer}
}
```

## Network
![architecture](https://github.com/hbliu98/figures/blob/main/1.png)

## Installation
The following environments/libraries are required
- pytorch
- SimpleITK
- yacs
- scikit-learn
- medpy
- torchio

## Usage
- For data preprocessing, please refer to [nnUNet](https://github.com/MIC-DKFZ/nnUNet), the preprocessed data should be placed in DATA/preprocessed folder
- run train.py / test.py
- Note that, the reconstructed images are loaded and augmented as the extra modality before sent into the segmentation network, so you can see code like this in Segmentation/core/function.py:
```
x, rec = data[:, :-1], data[:, -1:]
```
