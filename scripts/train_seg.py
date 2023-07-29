import gc
import math
import multiprocessing as mp
import os
import pprint
import uuid
from datetime import datetime
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import typer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

import wandb
from configs.factory import Config, init_config
from src.dataset import ContrailsDataset
from src.losses import get_loss
from src.metrics import calc_metrics
from src.models import ContrailsModel
from src.optimizer import get_optimizer
from src.scheduler import SchedulerType, get_scheduler
from src.train_tools import (
    AugParams,
    AuxParams,
    EarlyStopping,
    scheduler_step,
    seed_everything,
    train_one_epoch,
    valid_one_epoch,
)
from src.utils import add_file_handler, get_stream_logger

logger = get_stream_logger(20)

TODAY = datetime.today().strftime("%Y%m%d")


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
    # print(df)
    # print(df.iloc[0])
    # print(df.iloc[0]["record_id"])
    # print(df.iloc[0]["path"])
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
    print(df)
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    return train_df, valid_df


def get_loaders(
    config: Config, remake_df: bool, debug: bool, fold: int, positive_only: bool = False
) -> tuple[DataLoader, DataLoader]:
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

    num_workers = mp.cpu_count()
    if debug:
        train_df = train_df.sample(n=100, random_state=0)
        valid_df = valid_df.sample(n=100, random_state=0)
        num_workers = 1

    train_aug = A.Compose(config.train_aug_list)  # type: ignore
    valid_aug = A.Compose(config.valid_aug_list)  # type: ignore

    train_dataset = ContrailsDataset(
        df=train_df, image_size=config.image_size, train=True, transform_fn=train_aug
    )
    valid_dataset = ContrailsDataset(
        df=valid_df, image_size=config.image_size, train=True, transform_fn=valid_aug
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


def main(
    exp_ver: str, all: bool = False, debug: bool = False, remake_df: bool = False
) -> None:
    """
    Args:
        exp_ver: experiment version (e.g. exp000, exp001, ...)
        all: If True, train all folds
        debug: If True, train with debug mode
        remake_df: If True, remake dataframe
    """
    config_path = f"configs.{exp_ver}"
    config: Config = init_config(Config, config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    uid = str(uuid.uuid4())[:8]
    log_file_path = config.output_dir / f"train-{TODAY}-{uid}.log"
    add_file_handler(logger, str(log_file_path))

    logger.info(f"config_path: {config_path}")
    logger.info(f"{config.__dict__}")

    train_fold = list(range(config.n_splits)) if all else [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = True

    for fold in range(config.n_splits):
        if fold not in train_fold:
            continue
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
        )
        seed_everything(config.seed)
        logger.info(f"## Fold: {fold} ##")

        train_loader, valid_loader = get_loaders(
            config,
            remake_df=remake_df,
            debug=debug,
            fold=fold,
            positive_only=config.positive_only,
        )
        model = ContrailsModel(
            encoder_name=config.encoder_name,
            encoder_weight=config.encoder_weight,
            aux_params=config.aux_params,
            arch=config.arch,
        )
        model = model.to(device=device)
        if config.resume_training:
            resume_path = config.resume_path.format(fold=fold)
            logger.info(
                f"Resume training from {resume_path} with {config.positive_only = }"
            )
            state = torch.load(resume_path)
            model.load_state_dict(state)

        optimizer = get_optimizer(
            optimizer_type=config.optimizer_type,
            optimizer_params=config.optimizer_params,
            model=model,
        )

        schedule_per_step = False
        if SchedulerType(config.scheduler_type) == SchedulerType.CosineWithWarmup:
            step_num = (
                len(train_loader) // config.train_batch_size if schedule_per_step else 1
            )
            total_step_num = math.ceil(step_num * max(config.epochs, 1))
            warmup_step_num = (
                # math.ceil((total_step_num * 2) / 100)
                int(config.scheduler_params["warmup_ratio"] * total_step_num)
                if config.scheduler_params["warmup_ratio"]
                else 0
            )
            logger.info(f"{total_step_num = }, {warmup_step_num = }")
            scheduler_params: dict[str, int | float] = {
                "num_warmup_steps": warmup_step_num,
                "num_training_steps": total_step_num,
            }
        else:
            scheduler_params = config.scheduler_params
        scheduer = get_scheduler(
            scheduler_type=config.scheduler_type,
            scheduler_params=scheduler_params,
            optimizer=optimizer,
        )
        loss = get_loss(loss_type=config.loss_type, loss_params=config.loss_params)
        cls_loss = torch.nn.BCEWithLogitsLoss()
        scaler = GradScaler(enabled=use_amp)
        earlystopping = EarlyStopping(
            patience=config.patience,
            save_dir=config.output_dir,
            verbose=True,
            logger_fn=logger.info,
        )
        aux_params = (
            AuxParams(cls_weight=config.cls_weight) if config.cls_weight else None
        )
        aug_params = AugParams(**config.aug_params) if config.aug_params else None

        for epoch in range(config.epochs):
            seed_everything(config.seed)
            train_assets = train_one_epoch(
                fold=fold,
                epoch=epoch,
                model=model,
                train_loader=train_loader,
                criterion=loss,
                optimizer=optimizer,
                scheduler=scheduer,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                criterion_cls=cls_loss,
                aux_params=aux_params,
                aug_params=aug_params,
                schedule_per_step=schedule_per_step,
                max_grad_norm=config.max_grad_norm,
            )
            valid_assets = valid_one_epoch(
                fold=fold,
                epoch=epoch,
                model=model,
                valid_loader=valid_loader,
                criterion=loss,
                device=device,
                metrics_fn=calc_metrics,
                use_amp=use_amp,
                debug=debug,
            )
            scheduler_step(scheduer, valid_assets.loss)

            logging_assets = {
                "train/avg_loss": train_assets.loss,
                "train/avg_cls_acc": train_assets.cls_acc,
                "valid/avg_loss": valid_assets.loss,
                "valid/avg_dice": valid_assets.dice,
            }
            logger.info(f"{epoch}: \n{pprint.pformat(logging_assets)}")
            wandb.log(logging_assets)

            save_path = (
                f"{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}.pth"
            )
            earlystopping(valid_assets.dice, model, save_path)
            if earlystopping.early_stop:
                logger.info("Early stopping")
                break

        save_path = (
            f"last-{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}.pth"
        )
        earlystopping.save_checkpoint(float("inf"), model, config.output_dir / save_path)
        logger.info(f"## Fold: {fold} End ##")

        if run is not None:
            run.finish()

        gc.collect()
        torch.cuda.empty_cache()
    logger.info(f"## All End. Log -> {log_file_path} ##")


if __name__ == "__main__":
    typer.run(main)
