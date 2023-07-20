from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2

root = "."
expname = __file__.split("/")[-1].split(".")[0]

IMG_SIZE = 256
# IMG_SIZE = 512
# IMG_SIZE = 768
# IMG_SIZE = 1024

DESC = f"""
# exp009_x

## Purpose

make sure that cutmix works for bigger models with img_size={IMG_SIZE}

"""

config = {
    "expname": expname,
    "description": DESC,
    "seed": 42,
    # -- Model
    "arch": "UNet",
    # ref: https://smp.readthedocs.io/en/latest/encoders.html
    # "encoder_name": "timm-resnest26d",  # 16M
    # "encoder_name": "timm-resnest50d",  # 25M
    # "encoder_name": "timm-resnest101e",  # 46M
    "encoder_name": "timm-resnest200e",  # 86M
    "encoder_weight": "imagenet",
    "checkpoints": ["./output/exp009_8/exp009_8-UNet-timm-resnest26d-fold0.pth"],
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
    "image_size": IMG_SIZE,
    "n_splits": 5,
    # -- Training
    "train_batch_size": 16,
    "valid_batch_size": 32,
    "output_dir": Path(f"./output/{expname}"),
    "resume_training": True,
    "resume_path": "./output/exp009_9/exp009_9-UNet-timm-resnest200e-fold0.pth",
    "positive_only": True,
    "epochs": 50,
    "train_params": {},
    "max_grad_norm": 10.0,
    "patience": 12,
    # "loss_type": "soft_bce",
    # "loss_params": {"smooth_factor": 0.1},
    "loss_type": "dice",
    "loss_params": {"smooth": 1.0, "mode": "binary"},
    "cls_weight": 0.0,
    "aux_params": None,
    # "aux_params": {"dropout": 0.5, "classes": 1},
    "optimizer_type": "adamw",
    "optimizer_params": {
        "lr": 1e-5,
        "encoder_lr": 1e-5,
        "decoder_lr": 1e-4,
        "weight_decay": 0.0,
    },
    # "scheduler_type": "cosineannealinglr",
    # "scheduler_params": {
    #     "T_max": 2,  # iterationの最大(周期の長さ)
    #     "eta_min": 1e-6,  # lr最小値
    #     "last_epoch": -1,
    # },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_ratio": 0.02,
    },
    "train_aug_list": [
        # A.RandomResizedCrop(height=IMG_SIZE, width=IMG_SIZE, scale=(0.8, 1.2), p=1.0),
        # A.CropNonEmptyMaskIfExists(height=512, width=512, p=1.0),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1),
        # A.RandomRotate90(p=0.5),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
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
