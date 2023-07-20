from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

root = "."
expname = __file__.split("/")[-1].split(".")[0]

IMG_SIZE = 512

DESC = f"""
# exp013

About
-----

- use resnest50d

- use {IMG_SIZE} x {IMG_SIZE} images


"""

config = {
    "expname": expname,
    "description": DESC,
    "seed": 42,
    # -- Model
    "arch": "UNet",
    "encoder_name": "timm-resnest50d",  # 46M
    "encoder_weight": "imagenet",
    "checkpoints": ["./output/exp013/exp013-UNet-timm-resnest50d-fold0.pth"],
    # -- Data
    "data_root_path": Path(
        # f"{root}/input/google-research-identify-contrails-reduce-global-warming"
        f"{root}/input/contrails-images-ash-color"
    ),
    "train_csv_path": Path(
        f"{root}/input/google-research-identify-contrails-reduce-global-warming/train.csv"
    ),
    "valid_csv_path": Path(
        f"{root}/input/google-research-identify-contrails-reduce-global-warming/valid.csv"
    ),
    "image_size": 512,
    "n_splits": 5,
    # -- Training
    "train_batch_size": 16,
    "valid_batch_size": 32,
    "output_dir": Path(f"./output/{expname}"),
    "resume_training": False,
    "resume_path": "./output/exp013/exp013-UNet-timm-resnest50d-fold{fold}.pth",
    "positive_only": False,
    "epochs": 50,
    "train_params": {},
    "patience": 12,
    "loss_type": "dice",
    "loss_params": {"smooth": 1e-6, "mode": "binary"},
    "cls_weight": 0.0,
    "aux_params": None,
    "optimizer_type": "adamw",
    "optimizer_params": {
        "lr": 5e-4,
        "weight_decay": 0,
    },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_ratio": 0.03,
    },
    "train_aug_list": [
        # A.RandomResizedCrop(height=512, width=512, scale=(0.9, 1.0), p=1.0),
        # A.CropNonEmptyMaskIfExists(height=512, width=512, p=1.0),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
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
        do_cutmix=True,
        cutmix_alpha=1.0,
        cutmix_prob=0.5,
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
    config_ver = "exp002"
    pprint.pprint(init_config(Config, f"configs.{config_ver}"))
