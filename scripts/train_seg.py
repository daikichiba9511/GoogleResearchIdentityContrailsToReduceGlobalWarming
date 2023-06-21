import multiprocessing as mp
import pprint
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader

import wandb
from configs.factory import Config, init_config
from src.dataset import ContrailsDataset
from src.losses import get_loss
from src.metrics import calc_metrics
from src.models import ContrailsModel
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src.train_tools import (
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
    filenames = list(image_root_path.rglob("*"))
    df = pd.DataFrame(filenames, columns=["record_id"])
    df["path"] = data_root_path / phase / df["record_id"].astype(str)
    return df


def get_dfs(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    # contrails = config.data_root_path / "contrails"
    train_path = config.data_root_path / "train"
    valid_path = config.data_root_path / "valid"
    train_df = pd.read_csv(config.data_root_path / "train.csv")
    train_df["path"] = train_path + train_df["record_id"].astype(str) + ".npy"
    valid_df = pd.read_csv(config.data_root_path / "valid.csv")
    valid_df["path"] = valid_path + valid_df["record_id"].astype(str) + ".npy"
    return train_df, valid_df


def get_loaders(
    config: Config, remake_df: bool, debug: bool
) -> tuple[DataLoader, DataLoader]:
    if remake_df:
        train_df = make_df(
            config.data_root_path, config.data_root_path / "train", "train"
        )
        valid_df = make_df(
            config.data_root_path, config.data_root_path / "valid", "valid"
        )
    else:
        train_df, valid_df = get_dfs(config)

    if debug:
        train_df = train_df.sample(n=100, random_state=0)
        valid_df = valid_df.sample(n=100, random_state=0)

    train_dataset = ContrailsDataset(
        df=train_df, image_size=config.image_size, train=True
    )
    valid_dataset = ContrailsDataset(
        df=valid_df, image_size=config.image_size, train=True
    )
    num_workers = mp.cpu_count()
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
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader


def main(exp_ver: str, all: bool = False, debug: bool = False) -> None:
    """
    Args:
        exp_ver: experiment version (e.g. exp000, exp001, ...)
        all: If True, train all folds
        debug: If True, train with debug mode
    """
    config_path = f"configs.{exp_ver}"
    logger.info(f"config_path: {config_path}")
    config: Config = init_config(Config, config_path)
    logger.info(f"{config.__dict__}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    uid = str(uuid.uuid4())[:8]
    add_file_handler(logger, config.output_dir / f"train-{TODAY}-{uid}.log")

    train_fold = list(config.n_splits) if all else [0]
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

        train_loader, valid_loader = get_loaders(config, remake_df=True, debug=debug)
        model = ContrailsModel(
            encoder_name=config.encoder_name, encoder_weight=config.encoder_weight
        )
        optimizer = get_optimizer(
            optimizer_type=config.optimizer_type,
            optimizer_params=config.optimizer_params,
            model=model,
        )
        scheduler_params = {
            "num_warmup_steps": int(
                len(train_loader) * config.scheduler_params["warmup_step_ratio"]
            ),
            "num_training_steps": len(train_loader),
        }
        scheduer = get_scheduler(
            scheduler_type=config.scheduler_type,
            scheduler_params=scheduler_params,
            optimizer=optimizer,
        )
        loss = get_loss(loss_type=config.loss_type, loss_params=config.loss_params)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        earlystopping = EarlyStopping(
            patience=config.patience, save_dir=config.output_dir, verbose=True
        )

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
            )
            valid_assets = valid_one_epoch(
                fold=fold,
                epoch=epoch,
                model=model,
                valid_loader=valid_loader,
                loss=loss,
                device=device,
                metrics_fn=calc_metrics,
                use_amp=use_amp,
            )
            scheduler_step(scheduer, valid_assets["loss"], epoch=epoch)

            logging_assets = {
                "train/avg_loss": train_assets["loss"],
                "valid/avg_loss": valid_assets["loss"],
                "valid/avg_dice": valid_assets["dice"],
            }
            logger.info(f"{epoch}: \n{pprint.pformat(logging_assets)}")
            wandb.log(logging_assets)

            save_path = f"{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}-epoch{epoch}.pth"
            earlystopping(valid_assets["dice"], model, save_path)
            if earlystopping.early_stop:
                logger.info("Early stopping")
                break

        save_path = f"last-{config.expname}-{config.arch}-{config.encoder_name}-fold{fold}-epoch{epoch}.pth"
        earlystopping.save_checkpoint(float("inf"), model, save_path)
        logger.info(f"## Fold: {fold} End ##")
        run.finish()
    logger.info("## All End ##")


if __name__ == "__main__":
    typer.run(main)
