import gc
import math
import multiprocessing as mp
import os
import pprint
import uuid
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import typer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

import wandb
from configs.factory import Config, init_config
from src.dataset import ContrailsDataset, ContrailsDatasetV2
from src.losses import get_loss
from src.metrics import calc_metrics
from src.models import ContrailsModel, CustomedUnet, UNETR_Segformer
from src.optimizer import get_optimizer
from src.scheduler import SchedulerType, get_scheduler
from src.train_tools import (
    AugParams,
    AuxParams,
    EarlyStopping,
    make_tta_model,
    scheduler_step,
    seed_everything,
    train_one_epoch,
    valid_one_epoch,
)
from src.utils import add_file_handler, get_called_time, get_stream_logger

logger = get_stream_logger(20)

TODAY = get_called_time()


def make_df(data_root_path: Path, image_root_path: Path, phase: str) -> pd.DataFrame:
    filenames = os.listdir(data_root_path / phase)
    paths = []
    for record_id in filenames:
        label = image_root_path / record_id / "human_pixel_masks.npy"
        for image_file_path in (image_root_path / record_id).glob("band_*.npy"):
            paths.append(
                {"record_id": record_id, "path": image_file_path, "label": label}
            )
    df = pd.DataFrame(paths)
    logger.info(f"\n{df.head(10)}")
    return df


def get_dfs(config: Config, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    # train_path = valid_path = str(config.data_root_path / "contrails") + "/"
    # train_path = config.data_root_path / "train"
    # valid_path = config.data_root_path / "validation"
    # train_df = pd.read_csv(config.data_root_path / "train_df.csv")
    # train_df["path"] = train_path + train_df["record_id"].astype(str) + ".npy"
    # valid_df = pd.read_csv(config.data_root_path / "valid_df.csv")
    # valid_df["path"] = valid_path + valid_df["record_id"].astype(str) + ".npy"

    # df.csv is made by scripts/make_fold.py
    df = pd.read_csv(config.data_root_path / "df.csv")
    logger.info(f"\n{df}")
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    logger.info(f"{len(train_df) = }, {len(valid_df) =}")
    return train_df, valid_df


def get_loaders(
    config: Config,
    remake_df: bool,
    debug: bool,
    fold: int,
    positive_only: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_aug = A.Compose(config.train_aug_list)  # type: ignore
    valid_aug = A.Compose(config.valid_aug_list)  # type: ignore
    num_workers = mp.cpu_count() if not debug else 1

    if remake_df:
        train_df = make_df(
            config.data_root_path, config.data_root_path / "train", "train"
        )
        valid_df = make_df(
            config.data_root_path, config.data_root_path / "validation", "validation"
        )
    elif positive_only:
        df = pd.read_csv(config.data_root_path / "df.csv")
        print(df)
        train_df = df.query("cls_label == 1 and fold != @fold").reset_index(drop=True)
        # valid_df = df.query("cls_label == 1 and fold == @fold").reset_index(drop=True)
        valid_df = df.query("fold == @fold").reset_index(drop=True)
    else:
        train_df, valid_df = get_dfs(config, fold=fold)

    train_image_paths = train_df["path"].to_numpy().tolist()
    valid_image_paths = valid_df["path"].to_numpy().tolist()

    if config.with_pseudo_label:
        paths = list(Path(config.pseudo_label_dir).glob("*"))
        logger.info(f"Use pseudo labeled data {len(paths) = }")
        train_image_paths += paths

    if debug:
        train_image_paths = train_image_paths[:100]
        valid_image_paths = valid_image_paths[:100]

    logger.info(f"{len(train_image_paths) = }, {len(valid_image_paths) = }")

    train_dataset = ContrailsDataset(
        image_paths=train_image_paths,
        image_size=config.image_size,
        train=True,
        transform_fn=train_aug,
    )
    valid_dataset = ContrailsDataset(
        image_paths=valid_image_paths,
        image_size=config.image_size,
        train=True,
        transform_fn=valid_aug,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader


def get_loaders_v2(
    config: Config,
    debug: bool,
    fold: int,
    positive_only: bool = False,
    use_soft_label: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_aug = A.Compose(config.train_aug_list)  # type: ignore
    valid_aug = A.Compose(config.valid_aug_list)  # type: ignore
    num_workers = mp.cpu_count() if not debug else 1

    df = pd.read_csv("./input/prepared_np_imgs_weight1/metadata.csv")
    train_df = df.query("is_train").reset_index(drop=True)
    valid_df = df.query("not is_train").reset_index(drop=True)
    if debug:
        train_df = train_df[:100]
        valid_df = valid_df[:100]

    train_image_paths = train_df["image_path"].to_numpy().tolist()
    valid_image_paths = valid_df["image_path"].to_numpy().tolist()

    train_mask_paths = train_df["mask_path"].to_numpy().tolist()
    train_avg_mask_paths = train_df["avg_mask_path"].to_numpy().tolist()
    valid_mask_paths = valid_df["mask_path"].to_numpy().tolist()

    if config.with_pseudo_label:
        paths = list(Path(config.pseudo_label_dir).glob("*"))
        logger.info(f"Use pseudo labeled data {len(paths) = }")
        train_image_paths += paths

    logger.info(f"{len(train_image_paths) = }, {len(valid_image_paths) = }")

    train_dataset = ContrailsDatasetV2(
        img_paths=train_image_paths,
        transform_fn=train_aug,
        phase="train",
        use_soft_label=use_soft_label,
        mask_paths=train_mask_paths,
        avg_mask_paths=train_avg_mask_paths,
    )
    valid_dataset = ContrailsDatasetV2(
        img_paths=valid_image_paths,
        transform_fn=valid_aug,
        phase="val",
        mask_paths=valid_mask_paths,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.valid_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader


def builded_model(config: Config, disable_compile: bool, fold: int) -> torch.nn.Module:
    logger.info(f"{config.arch =}")

    if config.arch == "UNETR_Segformer":
        model = UNETR_Segformer(img_size=config.image_size)
    elif config.arch == "CustomedUnet":
        model = CustomedUnet(
            name=config.encoder_name,
            pretrained=config.encoder_weight is not None,
            tta_type=None,
        )
    else:
        model = ContrailsModel(
            encoder_name=config.encoder_name,
            encoder_weight=config.encoder_weight,
            aux_params=config.aux_params,
            arch=config.arch,
        )

    if config.resume_training:
        resume_path = config.resume_path.format(fold=fold)
        logger.info(
            f"Resume training from {resume_path} with {config.positive_only = }"
        )
        state = torch.load(resume_path)
        model.load_state_dict(state)

    if disable_compile:
        return model
    return torch.compile(model)  # type: ignore


def _init_scheduler(
    config: Config,
    optimizer: torch.optim.Optimizer,
    schedule_per_step: bool,
    train_loader: DataLoader,
) -> torch.optim.lr_scheduler.LRScheduler:
    # if SchedulerType(config.scheduler_type) == SchedulerType.CosineWithWarmup:
    #     step_num = (
    #         len(train_loader) // config.train_batch_size if schedule_per_step else 1
    #     )
    #     total_step_num = math.ceil(step_num * max(config.epochs, 1))
    #     warmup_step_num = (
    #         # math.ceil((total_step_num * 2) / 100)
    #         int(config.scheduler_params["warmup_ratio"] * total_step_num)
    #         if config.scheduler_params["warmup_ratio"]
    #         else 0
    #     )
    #     logger.info(f"{total_step_num = }, {warmup_step_num = }")
    #     scheduler_params: dict[str, int | float] = {
    #         "num_warmup_steps": warmup_step_num,
    #         "num_training_steps": total_step_num,
    #     }
    # else:
    #     scheduler_params = config.scheduler_params
    scheduler_params = config.scheduler_params
    scheduer = get_scheduler(
        scheduler_type=config.scheduler_type,
        scheduler_params=scheduler_params,
        optimizer=optimizer,
    )
    return scheduer


def setuped_run(config: Config, exp_ver: str, fold: int, uid: str):
    run = wandb.init(
        project="contrails",
        name=f"{exp_ver}-{uid}-fold{fold}-{config.arch}-{config.encoder_name}",
        notes=config.description,
        config=config.__dict__,
        group=f"{exp_ver.split('_')[0]}",
        tags=[
            exp_ver.split("_")[0],
            f"fold{fold}",
            config.arch,
            config.encoder_name,
        ],
        settings=wandb.Settings(code_dir="./src"),
    )
    if not isinstance(run, wandb.sdk.wandb_run.Run):  # type: ignore
        raise ValueError("wandb.init() returns unexpected type")

    return run


def main(
    exp_ver: str,
    all: bool = False,
    debug: bool = False,
    remake_df: bool = False,
    disable_compile: bool = False,
) -> None:
    """
    Args:
        exp_ver: experiment version (e.g. exp000, exp001, ...)
        all: If True, train all folds
        debug: If True, train with debug mode
        remake_df: If True, remake dataframe
    """
    config_path = f"configs.{exp_ver}"
    config: Config = init_config(Config, config_path, debug)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    uid = str(uuid.uuid4())[:8]
    log_file_path = config.output_dir / f"train-{TODAY}-{uid}.log"
    add_file_handler(logger, str(log_file_path))

    logger.info(f"config_path: {config_path}")
    logger.info(f"{pprint.pformat(config.__dict__)}")

    train_fold = list(range(config.n_splits)) if all else [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.use_amp
    schedule_per_step = False

    for fold in range(config.n_splits):
        if fold not in train_fold:
            continue
        seed_everything(config.seed)
        logger.info(f"## Fold: {fold} ##")
        run = setuped_run(config, exp_ver, fold, uid)
        train_loader, valid_loader = get_loaders_v2(
            config, debug, fold, config.positive_only, config.use_soft_label
        )
        model = builded_model(config, disable_compile, fold).to(
            device=device, non_blocking=True
        )
        optimizer = get_optimizer(config.optimizer_type, config.optimizer_params, model)
        scheduer = _init_scheduler(config, optimizer, schedule_per_step, train_loader)
        loss = get_loss(loss_type=config.loss_type, loss_params=config.loss_params)
        cls_loss = torch.nn.BCEWithLogitsLoss()
        scaler = GradScaler(enabled=use_amp)
        earlystopping = EarlyStopping(
            config.patience, True, logger.info, config.output_dir
        )
        aux_params = (
            AuxParams(cls_weight=config.cls_weight) if config.cls_weight else None
        )
        aug_params = AugParams(**config.aug_params) if config.aug_params else None

        assets = []
        for epoch in range(config.epochs):
            seed_everything(config.seed)
            train_assets = train_one_epoch(
                fold=fold,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                loss_fn=loss,
                optimizer=optimizer,
                scheduler=scheduer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                cls_loss_fn=cls_loss,
                aux_params=aux_params,
                aug_params=aug_params,
                schedule_per_step=schedule_per_step,
                max_grad_norm=config.max_grad_norm,
                grad_accum_step_num=config.grad_accum_step_num,
                metrics_fn=calc_metrics,
            )
            valid_assets = valid_one_epoch(
                fold=fold,
                epoch=epoch,
                model=model,
                valid_loader=valid_loader,
                loss_fn=loss,
                device=device,
                metrics_fn=calc_metrics,
                use_amp=use_amp,
                debug=debug,
            )

            if not schedule_per_step:
                scheduler_step(scheduer, valid_assets.loss, epoch)

            logging_assets = {
                "train/avg_loss": train_assets.loss,
                "train/duration": train_assets.duration,
                "train/avg_dice": train_assets.dice,
                "valid/avg_loss": valid_assets.loss,
                "valid/avg_dice": valid_assets.dice,
                "valid/duration": valid_assets.duration,
                "valid/global_dice": valid_assets.global_dice,
            }
            assets.append(logging_assets)
            logger.info(f"{epoch}: \n{pprint.pformat(logging_assets)}")
            wandb.log(logging_assets)

            save_path = (
                f"{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}.pth"
            )
            earlystopping(valid_assets.global_dice, model, save_path)
            if earlystopping.early_stop:
                logger.info("Early stopping")
                break

        save_path = (
            config.output_dir
            / f"last-{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}.pth"
        )
        earlystopping.save_checkpoint(assets[-1]["valid/global_dice"], model, save_path)
        logger.info(f"## Fold: {fold} End ##")
        if run is not None:
            run.finish()
        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"## All End. Log -> {log_file_path} ##")


if __name__ == "__main__":
    import argparse

    # all: bool = False, debug: bool = False, remake_df: bool = False

    args = argparse.ArgumentParser()
    args.add_argument("exp_ver", type=str)
    args.add_argument("--debug", default=False, action="store_true")
    args.add_argument("--all", default=False, action="store_true")
    args.add_argument("--remake_df", default=False, action="store_true")
    args.add_argument("--disable_compile", default=False, action="store_true")
    # typer.run(main)
    main(**vars(args.parse_args()))
