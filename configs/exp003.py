from pathlib import Path

root = "."
expname = __file__.split("/")[-1].split(".")[0]

config = {
    "expname": expname,
    "description": f"{expname}: Contrails segmentation baseline with cls_head",
    "seed": 42,
    # -- Model
    "arch": "UNet",
    "encoder_name": "efficientnet-b0",
    "encoder_weight": "imagenet",
    "checkpoints": ["./output/exp003/exp003-UNet-resnet18-fold0.pth"],
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
    "image_size": 256,
    "n_splits": 5,
    # -- Training
    "train_batch_size": 32,
    "valid_batch_size": 32,
    "output_dir": Path(f"./output/{expname}"),
    "epochs": 20,
    "train_params": {
        "epochs": 10,
    },
    "patience": 5,
    "loss_type": "soft_bce",
    "loss_params": {
        "smooth_factor": 0.00,
    },
    "cls_weight": 0.01,
    "aux_params": {"classes": 1, "dropout": 0.3},
    "optimizer_type": "adamw",
    "optimizer_params": {
        "lr": 5e-4,
        "weight_decay": 1e-5,
    },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_step_ratio": 0.001,
    },
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
