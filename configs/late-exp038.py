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
# exp038

copy from exp037

# Purpose

- arch: Unet
- encoder: tu-tf_efficientnetv2_m
- img_size={IMG_SIZE}
- with_pseudo_label
- soft_bce
- try to use soft label
- lr=1e-4

# Reference

[1] https://smp.readthedocs.io/en/latest/encoders.html

"""

config = {
    "expname": expname,
    "description": DESC,
    "seed": 42,
    # -- Model
    "arch": "Unet",
    "encoder_name": "tu-tf_efficientnetv2_l",
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
    "train_batch_size": 8 * 2,
    "valid_batch_size": 32,
    "output_dir": Path(f"./output/{expname}"),
    "resume_training": False,
    "resume_path": "",
    "positive_only": False,
    "epochs": 40,
    "train_params": {},
    "max_grad_norm": 1000.0,
    "patience": 12,
    "grad_accum_step_num": 1,
    "loss_type": "soft_bce",
    "loss_params": {"smooth_factor": 0.0, "pos_weight": torch.tensor(7.31)},
    "cls_weight": 0.0,
    "aux_params": None,
    "optimizer_type": "adamw",
    "optimizer_params": {
        "lr": 1e-4,
        "weight_decay": 0.0,
        "eps": 1e-4,
    },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_ratio": 0.05,
    },
    "train_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        A.RandomRotate90(p=0.4),
        A.HorizontalFlip(p=0.1),
        ToTensorV2(),
    ],
    "valid_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        ToTensorV2(),
    ],
    "test_aug_list": [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
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
