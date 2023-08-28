import argparse
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
import torchvision.transforms.functional as F_t
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

import wandb
from configs.factory import Config, made_config
from src.augmentations import cutmix, label_noise, mixup
from src.dataset import fixed_offset_img
from src.losses import get_loss
from src.metrics import GlobalDice, MetricsFns
from src.models import builded_model
from src.optimizer import get_optimizer
from src.scheduler import get_scheduler
from src.train_tools import AverageMeter, make_tta_model, seed_everything
from src.utils import (
    add_file_handler,
    get_called_time,
    get_stream_logger,
    plot_for_debug,
)

logger = get_stream_logger(20)
TODAY = get_called_time()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], dict[str, float | int]]


@dataclass(frozen=True)
class AugParams:
    do_mixup: bool
    mixup_alpha: float
    mixup_prob: float

    do_cutmix: bool
    cutmix_alpha: float
    cutmix_prob: float
    turn_off_cutmix_epoch: int | None

    do_label_noise: bool
    label_noise_prob: float


def mixed_imgs(
    aug_params: AugParams | None, epoch: int, batch: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    images = batch["image"]
    target = batch["target"]

    if not (isinstance(images, torch.Tensor) and isinstance(target, torch.Tensor)):
        raise TypeError(f"{type(images) = }, {type(target) = }")

    is_mixed = False
    if (
        aug_params is not None
        and aug_params.do_cutmix
        and aug_params.turn_off_cutmix_epoch is not None
        and epoch <= aug_params.turn_off_cutmix_epoch
        and np.random.rand() <= aug_params.cutmix_prob
    ):
        images, target, _, _ = mixup(images, target, alpha=aug_params.mixup_alpha)
        is_mixed = True

    if (
        aug_params is not None
        and aug_params.do_mixup
        and np.random.rand() <= aug_params.mixup_prob
    ):
        images, target, _, _ = cutmix(images, target, alpha=aug_params.cutmix_alpha)
        is_mixed = True

    if (
        aug_params is not None
        and aug_params.do_label_noise
        and np.random.rand() <= aug_params.label_noise_prob
    ):
        images, target, _ = label_noise(images, target)
        is_mixed = True

    mixed_batch = {
        "image": images,
        "target": target,
        "is_mixed": is_mixed,
    }
    return mixed_batch


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
        augment_prob: float = 0.8,
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
        self._augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self._image_dirs)

    @staticmethod
    def _rescale_range(img: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
        if bounds[0] >= bounds[1]:
            raise ValueError(f"bounds[0] must be less than bounds[1], but {bounds}")
        return (img - bounds[0]) / (bounds[1] - bounds[0])

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
        resized_img = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_LINEAR)
        return resized_img

    @classmethod
    def _get_false_color(cls, record_data: dict[str, np.ndarray]) -> np.ndarray:
        _t11_bounds = (243, 303)
        _cloud_tp_tdiff_bounds = (-4, 5)
        _tdiff_bounds = (-4, 2)

        band11 = record_data["band_11"]
        band14 = record_data["band_14"]
        band15 = record_data["band_15"]

        r = cls._rescale_range(band15 - band14, _tdiff_bounds)
        g = cls._rescale_range(band14 - band11, _cloud_tp_tdiff_bounds)
        b = cls._rescale_range(band14, _t11_bounds)

        # shape: (256, 256, T), T = n_times_before + n_times_after + 1 = 8
        false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        return false_color

    @staticmethod
    def _img2tensor(img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        image_dir = self._image_dirs[idx]
        record_id = self._record_ids[idx]

        ntimes_before = 4
        raw_image = (
            self._get_false_color(self._read_record(image_dir))[..., ntimes_before]
            .reshape(256, 256, 3)
            .astype(np.float32)
        )
        resize_fn = fixed_offset_img if self._fix_offset else self._resized_img
        resized_image = resize_fn(raw_image, self._img_size)

        if self._phase == "test":
            # resized_image = self._resized_img(raw_image, self._img_size)
            image = self._transform(image=resized_image)["image"]
            return {
                "image": image,
                "record_id": record_id,
            }

        pixel_mask: np.ndarray = (
            np.load(image_dir / "human_pixel_masks.npy")  # (256, 256, 1)
            .astype(np.float32)
            .reshape(256, 256)
        )
        if self._phase == "valid":
            assert pixel_mask.shape == (256, 256)
            # resized_image = self._resized_img(raw_image, self._img_size)
            augmented = self._transform(image=resized_image, mask=pixel_mask)
            return {
                "image": augmented["image"],
                "target": augmented["mask"].reshape(1, 256, 256),
                "record_id": record_id,
            }

        # --- Train phase
        avg_mask: np.ndarray = (
            np.load(image_dir / "human_individual_masks.npy")  # (256, 256, 1, R)
            .astype(np.float32)
            .mean(axis=-1)
            .reshape(256, 256)
        )
        mask = avg_mask if self._use_soft_label else pixel_mask
        assert mask.shape == (256, 256)

        if np.random.randn() < self._augment_prob:
            is_augmented = 1
            augmented = self._transform(image=resized_image, mask=mask)
            # (1, 256,256)
            raw_size_avg_mask: torch.Tensor = augmented["mask"].reshape(1, 256, 256)
            # (C, H, W)
            image = augmented["image"]
        else:
            is_augmented = 0
            # (1, 256, 256)
            raw_size_avg_mask = torch.tensor(mask).reshape(1, 256, 256)
            # (C, H, W)
            image = torch.from_numpy(resized_image).permute(2, 0, 1).float()

        # (1, H, W)
        resized_avg_mask = F_t.resize(
            raw_size_avg_mask,
            [*self._img_size],
            InterpolationMode.BILINEAR,
            antialias=True,
        ).reshape(1, *self._img_size)

        # 回転してないからtrainの精度はでない, 256x256
        target_pixel_mask = torch.from_numpy(pixel_mask).float().reshape(1, 256, 256)

        return {
            "image": image,
            "target": target_pixel_mask,
            "avg_mask": resized_avg_mask,
            "raw_size_avg_mask": raw_size_avg_mask,
            "record_id": record_id,
            "is_augmented": torch.tensor(is_augmented),
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
        A.Compose(train_aug, is_check_shapes=False),
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

    # Metrics Initialization
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
        loss = loss_fn(preds, target).mean(dim=(1, 2, 3)).mean()
        losses.update(loss.item(), bs)

        # Metrics
        metrics = metrics_fn((preds > thr).int(), target)
        global_dice.update((preds > thr).int(), target)
        dice.update(metrics["dice"], bs)
        accs.update(metrics["accs"], bs)

    return Metrics(
        global_dice.value, dice.avg, accs.avg, losses.avg, time.time() - start
    )


def _fit_one_fold(
    fold: int,
    config: Config,
    disable_compile: bool,
    debug: bool,
    save_assets: bool = False,
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
    patience_cnt = 0
    start = time.time()
    schedule_per_epoch = True

    for epoch in range(config.epochs):
        seed_everything(config.seed + fold)
        epoch_start = time.time()
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
            raw_size_avg_mask = batch["raw_size_avg_mask"].to(device, non_blocking=True)
            bs = image.size(0)
            # if augmented==True => 1 else 0
            is_augmented = batch["is_augmented"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast_mode.autocast(
                enabled=config.use_amp, dtype=torch.float16
            ):
                outs = model(image)
                loss_aug = loss_fn(outs["logits"], avg_mask).mean(dim=(1, 2, 3))
                loss_ori = loss_fn(outs["preds"], raw_size_avg_mask).mean(dim=(1, 2, 3))
                loss = torch.mean(loss_aug + is_augmented * loss_ori)

            # Backpropagation
            if config.use_amp:
                grad_scaler.scale(loss).backward()  # type: ignore
            else:
                loss.backward()

            # Metrics
            preds = (outs["preds"].detach().float().cpu() > config.threshold).int()
            train_metrics = MetricsFns.calc_metrics(preds, batch["target"])
            global_dice.update(preds, batch["target"])
            dice.update(train_metrics["dice"], bs)
            accs.update(train_metrics["accs"], bs)
            losses.update(loss.item(), bs)

            if (
                debug
                or save_assets
                or (epoch == config.epochs - 1 and i == len(dl_train) - 1)
                or (patience_cnt == config.patience - 1 and i == len(dl_train) - 1)
            ):
                save_dir = Path(f"debug/{config.expname}/train")
                save_dir.mkdir(parents=True, exist_ok=True)
                plot_for_debug(
                    image, batch["avg_mask"], preds, save_dir, batch["record_id"]
                )

            if (i + 1) % config.grad_accum_step_num == 0:
                # torch.nn.utils.clip_grad_norm_(  # type: ignore
                torch.nn.utils.clip_grad_value_(  # type: ignore
                    model.parameters(), config.max_grad_norm
                )

                if config.use_amp:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
            # --- End of train loop
        train_duration = time.time() - epoch_start
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

        # Early stopping
        score = metrics.global_dice
        if score > best_score:
            patience_cnt = 0
            logger.info(f"Updated score {best_score} -> {score}")
            best_score = score
            torch.save(
                model.state_dict(),
                config.output_dir
                / f"{config.arch}-{config.encoder_name}-fold{fold}.pth",
            )
        else:
            patience_cnt += 1
            logger.info(
                f"Score {score} did not improve from {best_score} for {patience_cnt}."
            )

        # Logging
        logger.info(
            f"""\n
                Metrics report
                --------------------------------

                Epoch               : {epoch}
                Total Time          : {time.time() - start}
                LR                  : {optimizer.param_groups[0]['lr']}

                [Train] Duration    : {train_duration}
                [Train] Loss        : {losses.avg}
                [Train] Avg Dice    : {dice.avg}
                [Train] Avg Accs    : {accs.avg}
                [Train] Global Dice : {global_dice.value}

                [Valid] Duration    : {metrics.duration}
                [Valid] Loss        : {metrics.loss}
                [Valid] Avg dice    : {metrics.batch_avg_dice}
                [Valid] Avg accs    : {metrics.batch_avg_acc}
                [Valid] Global Dice : {metrics.global_dice}
        """
        )
        wandb.log(
            {
                "lr": optimizer.param_groups[0]["lr"],
                "train/loss": losses.avg,
                "train/duaration": train_duration,
                f"train/fold{fold}_dice_avg": dice.avg,
                f"train/fold{fold}_accs_avg": accs.avg,
                f"train/fold{fold}_global_dice": global_dice.value,
                "valid/duration": metrics.duration,
                "valid/loss": metrics.loss,
                f"valid/fold{fold}_dice_avg": metrics.batch_avg_dice,
                f"valid/fold{fold}_accs_avg": metrics.batch_avg_acc,
                f"valid/fold{fold}_global_dice": metrics.global_dice,
            }
        )

        if config.patience <= patience_cnt:
            logger.info(f"Early stopping at epoch {epoch} with best score {best_score}")
            break
    # --- End of epoch loop
    torch.save(
        model.state_dict(),
        config.output_dir / f"last-{config.arch}-{config.encoder_name}-fold{fold}.pth",
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
        seed_everything(config.seed + fold)
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
