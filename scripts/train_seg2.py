import argparse
import importlib
import multiprocessing as mp
import pprint
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb
from configs.factory import Config, made_config
from src.dataset import fixed_offset_img
from src.losses import get_loss
from src.metrics import GlobalDice, MetricsFns
from src.models import builded_model
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src.train_tools import AverageMeter, make_tta_model
from src.utils import add_file_handler, get_called_time, get_stream_logger

logger = get_stream_logger(20)
TODAY = get_called_time()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], dict[str, float | int]]


##################
# Metrics
##################
@dataclass
class Metrics:
    global_dice: float
    batch_avg_dice: float
    batch_avg_acc: float
    loss: float
    duration: float


class ContrailDatasetV3(Dataset):
    def __init__(
        self,
        image_dirs: Sequence[Path],
        record_ids: Sequence[str],
        phase: str,
        transform: A.Compose,
        img_size: tuple[int, int] = (512, 512),
        fix_offset: bool = False,
        use_soft_label: bool = False,
    ) -> None:
        if phase not in ["train", "valid", "test"]:
            raise ValueError(
                f"phase must be one of ['train', 'valid', 'test'], but {phase} was given"
            )
        self._phase = phase
        self._record_ids = record_ids
        self._image_dirs = image_dirs
        self._transform = transform
        self._img_size = img_size
        self._fix_offset = fix_offset
        self._use_soft_label = use_soft_label

    def __len__(self) -> int:
        return len(self._image_dirs)

    @staticmethod
    def rescale_range(
        img: np.ndarray, min_value: float, max_value: float
    ) -> np.ndarray:
        return img - min_value / (max_value - min_value)

    @staticmethod
    def _read_record(
        dir: Path, load_band_files: list[str] = ["band_11", "band_14", "band_15"]
    ) -> dict[str, np.ndarray]:
        data = {}
        for band in load_band_files:
            data[band] = np.load(dir / f"{band}.npy")
        return data

    @staticmethod
    def _resized_img(img: np.ndarray, img_size: tuple[int, int]) -> np.ndarray:
        """
        Args:
            img: (H, W, C)
            img_size: (H, W)
        """
        resized_img = A.resize(
            img, height=img_size[0], width=img_size[1], interpolation=cv2.INTER_LINEAR
        )
        return resized_img

    @classmethod
    def _get_false_color(cls, record_data: dict[str, np.ndarray]) -> np.ndarray:
        _t11_bounds = (243, 303)
        _cloud_tp_tdiff_bounds = (-4, 5)
        _tdiff_bounds = (-4, 2)

        band11 = record_data["band_11"]
        band14 = record_data["band_14"]
        band15 = record_data["band_15"]

        r = cls.rescale_range(band15 - band14, _tdiff_bounds[0], _tdiff_bounds[1])
        g = cls.rescale_range(
            band14 - band11, _cloud_tp_tdiff_bounds[0], _cloud_tp_tdiff_bounds[1]
        )
        b = cls.rescale_range(band11, _t11_bounds[0], _t11_bounds[1])

        # shape: (256, 256, T), T = n_times_before + n_times_after + 1 = 8
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        return false_color

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_dir = self._image_dirs[idx]
        record_id = self._record_ids[idx]

        raw_image = (
            self._get_false_color(self._read_record(image_dir))[..., 4]
            .reshape(256, 256, 3)
            .astype(np.float32)
        )
        if self._phase == "test":
            image = self._transform(image=raw_image)["image"]
            return {
                "image": image,
                "record_id": record_id,
            }

        pixel_mask = (
            np.load(image_dir / "human_pixel_masks.npy")  # (256, 256, 1)
            .reshape(256, 256)
            .astype(np.float32)
        )
        if self._phase == "valid":
            if self._fix_offset:
                resized_image = fixed_offset_img(raw_image, self._img_size)
            else:
                resized_image = self._resized_img(raw_image, self._img_size)
            augmented = self._transform(image=resized_image, mask=pixel_mask)
            return {
                "image": augmented["image"],
                "target": augmented["mask"].unsqueeze(0),
                "record_id": record_id,
            }

        # --- Train phase
        avg_mask = (
            np.load(image_dir / "human_individual_masks.npy")  # (256, 256, 1, R)
            .astype(np.float32)
            .mean(axis=-1)
        )
        if self._fix_offset:
            resized_image = fixed_offset_img(raw_image, self._img_size)
        else:
            resized_image = self._resized_img(raw_image, self._img_size)

        if self._use_soft_label:
            resized_mask = self._resized_img(avg_mask, self._img_size)
        else:
            resized_mask = self._resized_img(pixel_mask, self._img_size)

        augmented = self._transform(image=resized_image, mask=resized_mask)
        return {
            "image": augmented["image"],
            "target": pixel_mask.reshape(1, 256, 256),
            "avg_mask": augmented["mask"].reshape(1, *self._img_size),
            "record_id": record_id,
        }


def _builded_loaders(
    train_image_dir: Sequence[Path],
    valid_image_dir: Sequence[Path],
    train_bs_size: int,
    valid_bs_size: int,
    train_aug: Sequence[A.BasicTransform],
    valid_aug: Sequence[A.BasicTransform],
) -> tuple[DataLoader, DataLoader]:
    num_workers = mp.cpu_count()
    train_record_ids = [p.name for p in train_image_dir]
    valid_record_ids = [p.name for p in valid_image_dir]
    ds_train = ContrailDatasetV3(
        train_image_dir,
        train_record_ids,
        "train",
        A.Compose(train_aug),
        fix_offset=True,
        use_soft_label=True,
    )
    ds_valid = ContrailDatasetV3(
        valid_image_dir,
        valid_record_ids,
        "valid",
        A.Compose(valid_aug, is_check_shapes=False),
        fix_offset=True,
        use_soft_label=True,
    )
    dl_train = DataLoader(
        ds_train,
        batch_size=train_bs_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=valid_bs_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return dl_train, dl_valid


def validated(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LossFn,
    metrics_fn: MetricFn,
    device: torch.device,
    thr: float,
    use_amp: bool = False,
) -> Metrics:
    model.eval()
    global_dice = GlobalDice()
    losses = AverageMeter("loss")
    dice = AverageMeter("dice")
    accs = AverageMeter("accs")
    model = make_tta_model(model)

    start = time.time()
    for i, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True)
        target = batch["target"]
        bs = image.size(0)

        with torch.inference_mode(), torch.cuda.amp.autocast_mode.autocast(
            enabled=use_amp, dtype=torch.float16
        ):
            outs = model(image)
        preds = outs["preds"].detach().float().cpu()
        loss = loss_fn(preds, target)
        losses.update(loss.item(), bs)

        # Metrics
        metrics = metrics_fn((preds > thr).float(), target)
        global_dice.update((preds > thr).float(), target)
        dice.update(metrics["dice"], bs)
        accs.update(metrics["accs"], bs)

    return Metrics(
        global_dice.value, dice.avg, accs.avg, losses.avg, time.time() - start
    )


def _fit_one_fold(
    fold: int, config: Config, disable_compile: bool, debug: bool
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = builded_model(config, disable_compile, fold).to(device)
    optimizer = get_optimizer(config.optimizer_type, config.optimizer_params, model)
    scheduler = get_scheduler(config.scheduler_type, config.scheduler_params, optimizer)
    grad_scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=config.use_amp)
    loss_fn = get_loss(config.loss_type, config.loss_params)

    train_image_dir = list((config.data_root_path / "train").glob("*"))
    valid_image_dir = list((config.data_root_path / "validation").glob("*"))
    if debug:
        train_image_dir = train_image_dir[:100]
        valid_image_dir = valid_image_dir[:100]

    dl_train, dl_valid = _builded_loaders(
        train_image_dir,
        valid_image_dir,
        config.train_batch_size,
        config.valid_batch_size,
        config.train_aug_list,  # type: ignore
        config.valid_aug_list,  # type: ignore
    )

    losses = AverageMeter("loss")
    dice = AverageMeter("dice")
    accs = AverageMeter("accs")
    global_dice = GlobalDice()
    best_score = 0.0
    start = time.time()
    schedule_per_epoch = True

    for epoch in range(config.epochs):
        if schedule_per_epoch:
            scheduler.step(epoch)

        model.train()
        for i, batch in tqdm(
            enumerate(dl_train),
            total=len(dl_train),
            dynamic_ncols=True,
        ):
            if not schedule_per_epoch:
                scheduler.step(len(dl_train) * epoch + i)

            # train with soft label
            image = batch["image"].to(device, non_blocking=True)
            avg_mask = batch["avg_mask"].to(device, non_blocking=True)
            # target = batch["target"].to(device, non_blocking=True)
            bs = image.size(0)

            with torch.cuda.amp.autocast_mode.autocast(
                enabled=config.use_amp, dtype=torch.float16
            ):
                outs = model(image)
                logits = outs["logits"]
                loss = loss_fn(y_pred=logits, y_true=avg_mask)  # type: ignore
                # loss += loss_fn(outs["preds"], target)  # size: 256

            # Metrics
            preds = (outs["preds"].detach().float().cpu() > config.threshold).float()
            train_metrics = MetricsFns.calc_metrics(preds, batch["target"])
            global_dice.update(preds, batch["target"])
            dice.update(train_metrics["dice"], bs)
            accs.update(train_metrics["accs"], bs)
            losses.update(loss.item(), bs)

            # Backpropagation
            if config.use_amp:
                grad_scaler.scale(loss).backward()  # type: ignore
            else:
                loss.backward()

            if (i + 1) % config.grad_accum_step_num == 0:
                torch.nn.utils.clip_grad_norm_(  # type: ignore
                    model.parameters(), config.max_grad_norm
                )

                if config.use_amp:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
            # --- End of train loop
        # --- Validation
        metrics = validated(
            model,
            dl_valid,
            loss_fn,
            MetricsFns.calc_metrics,
            device,
            config.threshold,
            config.use_amp,
        )
        score = metrics.global_dice
        if score > best_score:
            logger.info(f"Updated score {best_score} -> {score}")
            best_score = score
            torch.save(
                model.state_dict(),
                config.output_dir
                / f"{config.arch}-{config.encoder_name}-fold{fold}.pth",
            )
        logger.info(
            f"""\n
            Metrics report
            --------------------------------

            Epoch              : {epoch}
            lr                 : {optimizer.param_groups[0]['lr']}

            [Train] loss       : {losses.avg}
            [Train] avg dice   : {dice.avg}
            [Train] avg accs   : {accs.avg}
            [Train] global_dice: {global_dice.value}

            [Valid] loss       : {metrics.loss}
            [Valid] avg dice   : {metrics.batch_avg_dice}
            [Valid] avg accs   : {metrics.batch_avg_acc}
            [Valid] global_dice: {metrics.global_dice}
        """
        )
        wandb.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "train/loss": losses.avg,
                f"train/fold{fold}_dice_avg": dice.avg,
                f"train/fold{fold}_accs_avg": accs.avg,
                f"train/fold{fold}_global_dice": global_dice.value,
                "valid/loss": metrics.loss,
                f"valid/fold{fold}_dice_avg": metrics.batch_avg_dice,
                f"valid/fold{fold}_accs_avg": metrics.batch_avg_acc,
                f"valid/fold{fold}_global_dice": metrics.global_dice,
            }
        )
    logger.info(
        f"Finish training. best score: {best_score}, duration: {time.time() - start}"
    )


def main(
    config_ver: str,
    debug: bool = False,
    disable_compile: bool = False,
    train_all: bool = False,
) -> None:
    config = made_config(config_ver, debug)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    uid = str(uuid.uuid4())[:8]
    log_file_path = config.output_dir / f"train-{TODAY}-{uid}.log"
    add_file_handler(logger, str(log_file_path))

    run = wandb.init(
        project="contrails",
        name=f"{config.expname}-{config.arch}-{config.encoder_name}",
        notes=config.description,
        config=config.__dict__,
        group=f"{config.expname.split('_')[0]}",
        tags=[
            config.expname.split("_")[0],
            config.arch,
            config.encoder_name,
        ],
        settings=wandb.Settings(code_dir="./src"),
    )

    logger.info(
        pprint.pformat(
            {
                "config_ver": config_ver,
                "debug": debug,
                "disable_compile": disable_compile,
                **config.__dict__,
            }
        )
    )

    # --- Start training
    folds = config.n_splits if train_all else 1
    for fold in range(folds):
        logger.info(f"Start training fold {fold}...")
        _fit_one_fold(fold, config, disable_compile, debug)

    if run is not None:
        run.finish()
    logger.info(f"Finish {config.expname} log file: {log_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable_compile", action="store_true")
    parser.add_argument("--train_all", action="store_true")
    args = parser.parse_args()
    main(args.config, args.debug, args.disable_compile, args.train_all)
