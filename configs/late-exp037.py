from pathlib import Path

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

root = "."
expname = __file__.split("/")[-1].split(".")[0]

# IMG_SIZE = 256
IMG_SIZE = 512
# IMG_SIZE = 768
# IMG_SIZE = 1024

DESC = f"""
# exp037

copy from exp035_4

# Purpose

- encoder: tu-maxvit_tiny_tf_512
- img_size={IMG_SIZE}
- with_pseudo_label
- soft_bce
- try to use soft label
- lr=1e-4

# Log

1. warmup_ratio=0.1 -> 0.43
2. warmup_ratio=0.02 -> 0.0
3. warmup_ratio=0.1,lr=1e-4 -> 0.0
4. warmup_ratio=0.1,lr=1e-4,fix a bug of kernel_size=1 in Head -> 0.0
5. warmup_ratio=0.1,lr=5e-4,fix a bug of kernel_size=1 in Head -> 0.0
6. warmup_ratio=0.1,lr=1e-4,fix a bug of kernel_size=1 in Head, grad_accum=1 -> 0.0
7. warmup_ratio=0.1,lr=5e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31 -> 0.0
8. warmup_ratio=0.1,lr=8e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31 -> 0.0
9. warmup_ratio=0.1,lr=8e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31, max_grad_norm=10.0 -> 0.07 (epoch5)
10. warmup_ratio=0.1,lr=8e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31, max_grad_norm=1000.0 -> 0.266 (epoch7)
11. warmup_ratio=0.1,lr=1e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31, max_grad_norm=1000.0 -> 0.271 (epoch11)
warmup_ratio=0.1,lr=1e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31, max_grad_norm=1000.0, RoandomState90(p=1.0) -> 0.292 (epoch18)
warmup_ratio=0.05,lr=1e-4,fix a bug of kernel_size=1 in Head, grad_accum=1, pos_weight=7.31, max_grad_norm=1000.0, RoandomState90(p=1.0) -> 0.265 (epoch39)

## 12

- warmup_ratio=0.05
- lr=1e-4
- fix a bug of kernel_size=1 in Head
- grad_accum=1
- pos_weight=7.31
- max_grad_norm=1000.0
- RoandomState90(p=1.0)
- without vflip

score: 0.222 (epoch15)

## 13

retry 12 without RandomResizedCrop,CropNonEmptyMaskIfExists

- warmup_ratio=0.05
- lr=1e-4
- fix a bug of kernel_size=1 in Head
- grad_accum=1
- pos_weight=7.31
- max_grad_norm=1000.0
- RoandomState90(p=1.0)
- without vflip

score: 0.247 (epoch18)

## 14

without augmentation to avoid asymmetry in the mask due to rotation

- warmup_ratio=0.05
- lr=1e-4
- fix a bug of kernel_size=1 in Head
- grad_accum=1
- pos_weight=7.31
- max_grad_norm=1000.0
- without augmentation

score: 0.274 (epoch39)

## 14

turn off amp

- warmup_ratio=0.05
- lr=1e-4
- fix a bug of kernel_size=1 in Head
- grad_accum=1
- pos_weight=7.31
- max_grad_norm=1000.0
- without augmentation
- turn off amp

score: 0.274 (epoch39)

# Reference

[1] https://smp.readthedocs.io/en/latest/encoders.html

"""

config = {
    "expname": expname,
    "description": DESC,
    "seed": 42,
    # -- Model
    "arch": "CustomedUnet",
    "encoder_name": "maxvit_tiny_tf_512.in1k",
    "encoder_weight": "imagenet",
    "checkpoints": ["./output/exp009_8/exp009_8-UNet-timm-resnest26d-fold0.pth"],
    # -- Data
    "data_root_path": Path(
        f"{root}/input/google-research-identify-contrails-reduce-global-warming"
        # f"{root}/input/contrails-images-ash-color"
    ),
    "train_csv_path": Path(
        f"{root}/input/google-research-identify-contrails-reduce-global-warming/train.csv"
    ),
    "valid_csv_path": Path(
        f"{root}/input/google-research-identify-contrails-reduce-global-warming/valid.csv"
    ),
    "with_pseudo_label": False,
    "pseudo_label_dir": Path(f"{root}/input/imgs_with_pseudo_labels_20230730185409"),
    "image_size": IMG_SIZE,
    "n_splits": 5,
    # -- Training
    "use_soft_label": True,
    "use_amp": True,
    "train_batch_size": 16,
    "valid_batch_size": 32,
    "output_dir": Path(f"./output/{expname}"),
    "resume_training": False,
    "resume_path": "",
    "positive_only": False,
    "epochs": 40,
    "train_params": {},
    "max_grad_norm": 1000,
    "patience": 10,
    "grad_accum_step_num": 1,
    "loss_type": "soft_bce",
    "loss_params": {"smooth_factor": 0.0, "pos_weight": torch.tensor(7.31)},
    "cls_weight": 0.0,
    "aux_params": None,
    "optimizer_type": "adamw",
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 1e-2,
        "eps": 1e-4,
    },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_ratio": 0.025,
    },
    "train_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        # A.RandomRotate90(p=1.0),
        # A.HorizontalFlip(p=0.2),
        A.Normalize(max_pixel_value=1.0),
        ToTensorV2(),
    ],
    "valid_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        A.Normalize(max_pixel_value=1.0),
        ToTensorV2(),
    ],
    "test_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        A.Normalize(max_pixel_value=1.0),
        ToTensorV2(),
    ],
    "aug_params": dict(
        do_mixup=False,
        mixup_alpha=1.0,
        mixup_prob=0.5,
        do_cutmix=False,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
        turn_off_cutmix_epoch=6,
        do_label_noise=False,
        label_noise_prob=0.5,
    ),
    # -- Inference
    "test_batch_size": 32,
    "threshold": 0.5,
}


if __name__ == "__main__":
    import pprint

    from configs.factory import Config, init_config

    pprint.pprint(config)
    print("## init config ##")
    config_ver = "exp030"
    pprint.pprint(init_config(Config, f"configs.{config_ver}"))
