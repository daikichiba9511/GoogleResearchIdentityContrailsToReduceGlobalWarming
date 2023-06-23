from pathlib import Path

root = "."
expname = __file__.split("/")[-1].split(".")[0]

config = {
    "expname": expname,
    "description": f"{expname}: Contrails segmentation baseline",
    "seed": 42,
    # -- Model
    "arch": "UNet",
    "encoder_name": "resnet18",
    "encoder_weight": "imagenet",
    "checkpoints": ["./output/exp001/exp001-UNet-resnet18-fold0.pth"],
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
        "lr": 1e-5,
    },
    "patience": 5,
    "loss_type": "bce",
    "loss_params": {},
    "optimizer_type": "adamw",
    "optimizer_params": {
        "weight_decay": 1e-4,
    },
    "scheduler_type": "cosine_with_warmup",
    "scheduler_params": {
        "warmup_step_ratio": 0.01,
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
    config_ver = "exp001"
    pprint.pprint(init_config(Config, f"configs.{config_ver}"))
