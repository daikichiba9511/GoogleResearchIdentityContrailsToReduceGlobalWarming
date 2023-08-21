import os
import random
import time
from collections import namedtuple
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ttach as tta
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import src.utils as my_utils
import wandb
from src.augmentations import cutmix, label_noise, mixup
from src.losses import LossFn

logger = getLogger(__name__)


__all__ = [
    "train_one_epoch",
    "AverageMeter",
    "AuxParams",
    "AWPParams",
    "AugParams",
    "FreezeParams",
    "seed_everything",
    "scheduler_step",
]


class AWP:
    """Adversarial Weight Perturbation

    Args:
        model (torch.nn.Module): model
        optimizer (torch.optim.Optimizer): optimizer
        criterion (Callable): loss function
        adv_param (str): parameter name to be perturbed. Defaults to "weight".
        adv_lr (float): learning rate. Defaults to 0.2.
        adv_eps (int): epsilon. Defaults to 1.
        start_epoch (int): start epoch. Defaults to 0.
        adv_step (int): adversarial step. Defaults to 1.
        scaler (torch.cuda.amp.GradScaler): scaler. Defaults to None.

    Examples:
    >>> model = Model()
    >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    >>> batch_size = 16
    >>> epochs = 10
    >>> num_train_train_steps = int(len(train_images) / batch_size * epochs)
    >>> awp = AWP(
    ...    model=model,
    ...    optimizer=optimizer,
    ...    adv_lr=1e-5,
    ...    adv_eps=3,
    ...    start_epoch=num_train_steps // epochs,
    ...    scaler=None,
    ... )
    >>> awp.attack_backward(image, mask_label, epoch)

    References:
    1.
    https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf
    2.
    https://speakerdeck.com/masakiaota/kaggledeshi-yong-sarerudi-dui-xue-xi-fang-fa-awpnolun-wen-jie-shuo-toshi-zhuang-jie-shuo-adversarial-weight-perturbation-helps-robust-generalization
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: LossFn,
        adv_param: str = "weight",
        adv_lr: float = 0.2,
        adv_eps: int = 1,
        start_epoch: int = 0,
        adv_step: int = 1,
        scaler: GradScaler | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.start_epoch = start_epoch
        self.adv_step = adv_step
        self.backup: dict[str, torch.Tensor] = {}
        self.backup_eps: dict[str, Any] = {}
        self.scaler = scaler
        self.criterion = criterion
        self.enable_autocast = scaler is not None

    def attack_backward(self, x: torch.Tensor, y: torch.Tensor, epoch: int) -> None:
        if (self.adv_lr == 0) or (epoch < self.start_epoch):
            return None
        self._save()
        for i in range(self.adv_step):
            self._attack_step()
            with autocast(device_type=x.device.type, enabled=self.enable_autocast):
                logits = self.model(x)["ink"]
                adv_loss = self.criterion(logits, y)
                adv_loss = adv_loss.mean()

            if self.scaler is not None:
                scaled_loss = self.scaler.scale(adv_loss)
                if isinstance(scaled_loss, torch.Tensor):
                    scaled_loss.backward()
                else:
                    raise TypeError(
                        f"Expected torch.Tensor, but got {type(scaled_loss)}"
                    )
            else:
                adv_loss.backward()

            self.optimizer.zero_grad(set_to_none=True)

        self._restore()

    def _attack_step(self) -> None:
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self) -> None:
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}


class AverageMeter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    def __str__(self) -> str:
        return f"Metrics {self.name}: Avg {self.avg}, Std {self.std}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.std = 0.0
        self.sum = 0.0
        self.count = 0
        self.rows: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        if value in [np.nan, np.inf, -np.inf, float("inf"), float("-inf")]:
            logger.info("Skip nan or inf value")
            return None
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.std = float(np.std(self.rows)) if len(self.rows) > 0 else 0.0
        self.rows.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {
            "name": self.name,
            "avg": self.avg,
            "std": self.std,
            "row_values": self.rows,
        }


class EarlyStopping:
    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        logger_fn: Callable = print,
        save_dir: Path = Path("./output"),
        fold: str = "0",
        save_prefix: str = "",
        direction: str = "maximize",
    ) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False
        self.min_score = np.Inf
        self.delta = delta
        self.logger_fn = logger_fn
        self.fold = fold
        self.save_prefix = save_prefix
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.direction = direction

    def __call__(self, score: float, model: nn.Module, save_path: Path | str) -> None:
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                score, model=model, save_path=self.save_dir / save_path
            )

        is_min_update = (
            self.direction == "maximize" and score < self.best_score + self.delta
        )
        is_max_update = (
            self.direction == "minimize" and score > self.best_score + self.delta
        )
        if is_min_update or is_max_update:
            self.counter += 1
            self.logger_fn(
                f"EarlyStopping Counter: {self.counter} out of {self.patience}"
                + f" for fold {self.fold} with best score {self.best_score}"
            )
            if self.counter >= self.patience:
                self.early_stop = True

        # update best score
        # maximize: score >= self.best_score + self.delta
        # minimize: score <= self.best_score + self.delta
        else:
            self.logger_fn(
                f"Detected update Score: best score {self.best_score} --> {score}"
            )
            self.best_score = score
            self.save_checkpoint(
                score, model=model, save_path=self.save_dir / save_path
            )
            self.counter = 0

    def save_checkpoint(self, score: float, model: nn.Module, save_path: Path) -> None:
        """Save model when validation loss decrease."""
        if self.verbose:
            self.logger_fn(f"Updated Score: ({self.min_score} --> {score})")

        state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        self.min_score = score


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.autograd.anomaly_mode.set_detect_anomaly(False)


def _make_cls_label(mask: torch.Tensor) -> torch.Tensor:
    """make classification label from mask

    Args:
        mask (torch.Tensor): mask, (B, H, W)

    Returns:
        torch.Tensor: classification label
    """
    cls_label = mask.sum(dim=(1, 2)) > 0
    assert cls_label.shape == (mask.shape[0],)
    return cls_label.float()


def _freeze_model(model: nn.Module, freeze_keys: Iterable[str] = ["encoder"]) -> None:
    """freeze model parameters specified by freeze_keys

    Args:
        model (nn.Module): model
    """
    for name, param in model.named_parameters():
        contains = [key in name for key in freeze_keys]
        if any(contains):
            param.requires_grad = False


def scheduler_step(
    scheduler: optim.lr_scheduler.LRScheduler, loss: float, epoch: int | None = None
) -> None:
    if loss in [float("inf"), float("-inf"), None, torch.nan, float("nan")]:
        return

    if epoch is None:
        scheduler.step()
    else:
        scheduler.step(epoch=epoch)


@dataclass(frozen=True)
class AWPParams:
    adv_lr: float
    adv_eps: int
    start_epoch: int
    adv_step: int


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


@dataclass(frozen=True)
class AuxParams:
    cls_weight: float
    cls_threshold: float = 0.5


@dataclass(frozen=True)
class FreezeParams:
    start_epoch_to_freeze_model: int
    freeze_keys: list[str]


def remove_tiny_pred(pred_mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove tiny predicted mask

    Args:
        pred_mask (np.ndarray): predicted mask (dtype: np.uint8, shape: (H, W))
        min_size (int, optional): minimum size of predicted mask. Defaults to 10.

    Returns:
        np.ndarray: predicted mask
    """
    if pred_mask.ndim != 2:
        raise ValueError(f"{pred_mask.shape = }")

    pred = pred_mask.copy()
    ret, labels = cv2.connectedComponents(pred_mask)
    for label in range(1, ret):
        area = (labels == label).sum()
        if area < min_size:
            pred[labels == label] = 0

    return pred


def init_average_meters(metric_names: Sequence[str]) -> dict[str, AverageMeter]:
    return {metric_name: AverageMeter(name=metric_name) for metric_name in metric_names}


@dataclass(frozen=True)
class ForwardOutputs:
    preds: torch.Tensor
    targets: torch.Tensor
    logits: torch.Tensor | None = None
    raw_targets: torch.Tensor | None = None
    cls_preds: torch.Tensor | None = None


@dataclass(frozen=True)
class PostprocessOutputs:
    preds: np.ndarray
    targets: np.ndarray


class BatchAnnotation(TypedDict):
    image: torch.Tensor
    target: torch.Tensor


class MetricsFn(Protocol):
    """Metrics function protocol

    Signature: (preds: np.ndarray, target: np.ndarray) -> dict[str, float | int]
    """

    def __call__(self, preds: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
        ...


class ForwardFn(Protocol):
    """Forward function protocol

    Signature: (model: nn.Module, batch: BatchAnnotation) -> ForwardOutputs
    """

    def __call__(self, model: nn.Module, batch: BatchAnnotation) -> ForwardOutputs:
        ...


class PostProcessFn(Protocol):
    """Postprocess function protocol

    Signature: (preds: torch.Tensor, targets: torch.Tensor) -> PostprocessOutputs

    """

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> PostprocessOutputs:
        ...


class AugmentationFn(Protocol):
    """Augmentation function protocol

    Signature: (aug_params: AugParams | None, epoch: int, batch: BatchAnnotation) -> BatchAnnotation
    """

    def __call__(
        self, aug_params: AugParams | None, epoch: int, batch: BatchAnnotation
    ) -> BatchAnnotation:
        ...


def default_segmentation_forward_fn(
    model: torch.nn.Module, batch: BatchAnnotation
) -> ForwardOutputs:
    image = batch["image"]
    target = batch["target"]
    if not (isinstance(image, torch.Tensor) and isinstance(target, torch.Tensor)):
        raise TypeError(f"{type(image) = }, {type(target) = }")

    output = model(image)
    logits = output["logits"].squeeze(1)
    preds = output["preds"].squeeze(1)

    if target.shape[1:] != (256, 256):
        resized_target = F.interpolate(
            target.unsqueeze(1).float(), size=(256, 256), mode="bilinear"
        ).squeeze(1)
    else:
        resized_target = target.squeeze(1).float()

    assert (
        resized_target.shape == preds.shape
    ), f"{resized_target.shape = }, {preds.shape = }"

    return ForwardOutputs(
        preds=preds, targets=resized_target, logits=logits, raw_targets=target
    )


def default_augmentation_fn(
    aug_params: AugParams | None, epoch: int, batch: BatchAnnotation
) -> BatchAnnotation:
    images = batch["image"]
    target = batch["target"]
    if not (isinstance(images, torch.Tensor) and isinstance(target, torch.Tensor)):
        raise TypeError(f"{type(images) = }, {type(target) = }")

    if (
        aug_params is not None
        and aug_params.do_cutmix
        and aug_params.turn_off_cutmix_epoch is not None
        and epoch <= aug_params.turn_off_cutmix_epoch
        and np.random.rand() <= aug_params.cutmix_prob
    ):
        images, target, _, _ = mixup(images, target, alpha=aug_params.mixup_alpha)

    if (
        aug_params is not None
        and aug_params.do_mixup
        and np.random.rand() <= aug_params.mixup_prob
    ):
        images, target, _, _ = cutmix(images, target, alpha=aug_params.cutmix_alpha)

    if (
        aug_params is not None
        and aug_params.do_label_noise
        and np.random.rand() <= aug_params.label_noise_prob
    ):
        images, target, _ = label_noise(images, target)

    return {"image": images, "target": target}


_T = TypeVar("_T", bound=BatchAnnotation)


def send_tensor_to_device(batch: _T, device: torch.device) -> _T:
    new_batch: _T = {}  # type: ignore
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            new_batch[key] = value.to(device, non_blocking=True)
        else:
            new_batch[key] = value
    return new_batch


def default_postprocess_fn(
    preds: torch.Tensor, targets: torch.Tensor
) -> PostprocessOutputs:
    # make a whole image prediction
    # y_preds: (N, H, W), target: (N, H, W)

    y_preds = preds.sigmoid().cpu().detach().numpy()
    target = targets.cpu().detach().numpy()

    assert isinstance(y_preds, np.ndarray)
    assert isinstance(target, np.ndarray)

    return PostprocessOutputs(preds=y_preds, targets=target)


def _init_awp(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    loss_fn: LossFn,
    awp_params: AWPParams | None = None,
) -> AWP | None:
    if awp_params is None:
        return None

    return AWP(
        model=model,
        optimizer=optimizer,
        criterion=loss_fn,
        scaler=scaler,
        adv_lr=awp_params.adv_lr,
        adv_eps=awp_params.adv_eps,
        start_epoch=awp_params.start_epoch,
        adv_step=awp_params.adv_step,
    )


class TrainAssets(NamedTuple):
    loss: float
    duration: int
    cls_acc: float | None = None


def train_one_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: LossFn,
    scaler: GradScaler,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    metrics_fn: MetricsFn,
    metric_names: list[str] = ["loss", "dice"],
    max_grad_norm: float = 1000.0,
    cls_loss_fn: LossFn | None = None,
    schedule_per_step: bool = False,
    use_amp: bool = False,
    grad_accum_step_num: int = 1,
    awp_params: AWPParams | None = None,
    aug_params: AugParams | None = None,
    augmentation_fn: AugmentationFn = default_augmentation_fn,
    aux_params: AuxParams | None = None,
    freeze_params: FreezeParams | None = None,
    forward_fn: ForwardFn = default_segmentation_forward_fn,
) -> TrainAssets:
    """Train one epoch

    Args:
        fold (int): fold number

        epoch (int): epoch number

        model (nn.Module): model

        train_loader (DataLoader): train data loader

        loss_fn (LossFn): loss function

        scaler (GradScaler): grad scaler

        optimizer (optim.Optimizer): optimizer

        scheduler (optim.lr_scheduler.LRScheduler): scheduler

        device (torch.device): device

        max_grad_norm (float, optional): max grad norm. Defaults to 1000.0.

        cls_loss_fn (LossFn, optional): classification loss function. Defaults to None.

        schedule_per_step (bool, optional): whether to schedule per step. Defaults to False.

        use_amp (bool, optional): whether to use amp. Defaults to False.

        grad_accum_step_num (int, optional): grad accumulation step number. Defaults to 1.

        awp_params (AWPParams, optional): awp params. Defaults to None.

        aug_params (AugParams, optional): augmentation params. Defaults to None.

        augmentation_fn (AugmentationFn, optional): augmentation function. Defaults to default_augmentation_fn. Signature: (aug_params: AugParams, epoch: int, batch: T<:BatchAnnotation) -> T<:BatchAnnotation.

        aux_params (AuxParams, optional): aux params. Defaults to None.

        freeze_params (FreezeParams, optional): freeze params. Defaults to None.

        forward_fn (ForwardFn, optional): forward function. Defaults to default_segmentation_forward_fn. Signature: (model: nn.Module, batch: T<:BatchAnnotation>) -> ForwardOutputs.

        metric_names (list[str], optional): metric names. Defaults to ["loss"].

    Returns:
        TrainAssets: train assets
    """
    # used for freeze model
    is_frozen = False

    start = time.time()

    # --- set up average meters for logging and evaluation
    if "loss" not in metric_names:
        metric_names.append("loss")
    if aux_params is not None and "aux_loss" not in metric_names:
        metric_names.append("aux_loss")
    average_meters = init_average_meters(metric_names=metric_names)
    model.train()

    pbar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
        desc="Train Per Epoch",
    )
    for step, batch in pbar:
        optimizer.zero_grad(set_to_none=True)

        # TODO: Dataset修正して書き直す
        if isinstance(batch, tuple):
            batch = {"image": batch[0], "target": batch[1]}

        batch = augmentation_fn(aug_params=aug_params, epoch=epoch, batch=batch)
        batch = send_tensor_to_device(batch, device=device)
        batch_size = batch["target"].size(0)

        if (
            not is_frozen
            and freeze_params is not None
            and epoch > freeze_params.start_epoch_to_freeze_model
        ):
            logger.info(f"freeze model with {freeze_params.freeze_keys}")
            _freeze_model(model, freeze_keys=freeze_params.freeze_keys)
            is_frozen = True  # cache state of freeze

        # --- Forward
        image = batch["image"]
        target = batch["target"]
        with autocast(device_type=device.type, enabled=use_amp):
            # output = forward_fn(model=model, batch=batch)
            outs = model(image)

        logits = outs["logits"].squeeze(1)
        preds = outs["preds"].squeeze(1)
        if target.shape[1:] != (256, 256):
            resized_target = F.interpolate(
                target.unsqueeze(1).float(), size=(256, 256), mode="bilinear"
            ).squeeze(1)
        else:
            resized_target = target.squeeze(1).float()

        output = ForwardOutputs(
            preds=preds, targets=resized_target, logits=logits, raw_targets=target
        )

        if output.logits is None or output.raw_targets is None:
            raise ValueError(f"{output.logits = }, {output.raw_targets = }")

        # --- Signature of loss_fn: (torch.Tensor, torch.Tensor) -> torch.Tensor
        # loss = loss_fn(output.logits, output.raw_targets)
        loss = loss_fn(output.preds, output.targets)

        # --- Aux task
        if (
            aux_params is not None
            and output.cls_preds is not None
            and cls_loss_fn is not None
        ):
            cls_preds = output.cls_preds
            cls_targets = _make_cls_label(output.targets)
            cls_loss = cls_loss_fn(cls_preds, cls_targets)
            cls_loss = aux_params.cls_weight * cls_loss
            loss += cls_loss

            cls_preds = (cls_preds > aux_params.cls_threshold).long()
            acc = (cls_preds == cls_targets).sum() / batch_size

            wandb.log({"train/cls_loss": cls_loss.item(), "train/cls_acc": acc.item()})

        loss = loss / min(grad_accum_step_num, 1)
        average_meters["loss"].update(value=loss.item(), n=batch_size)

        # --- Metrics
        preds = output.preds.sigmoid().cpu().detach().numpy()
        targets = output.targets.cpu().detach().numpy()
        train_metrics = metrics_fn(preds=preds, target=targets)
        train_metrics["loss"] = loss.item()
        for name, metric_value in train_metrics.items():
            average_meters[name].update(value=metric_value, n=batch_size)

        # --- Backprop
        scaled_loss = scaler.scale(loss)
        if not isinstance(scaled_loss, torch.Tensor):
            raise ValueError(f"Not Expected {scaled_loss = }, {type(scaled_loss) = }")

        scaled_loss.backward()

        if (step + 1) % grad_accum_step_num == 0:
            # Since not change gradient direction, usually use thie function
            clip_grad_norm_(model.parameters(), max_grad_norm)
            # clip_grad_value_(model.parameters(), max_grad_norm)  #

            # update parameters
            scaler.step(optimizer)
            scaler.update()

            if schedule_per_step:
                scheduler_step(scheduler, loss=loss.item())

            def _make_logging_assets(name: str) -> dict[str, float]:
                return {
                    f"train/fold{fold}_{name}_avg": average_meters[name].avg,
                    # f"train/fold{fold}_{name}_std": average_meters[name].std,
                }

            log_assets = list(map(_make_logging_assets, metric_names))
            log_assets = my_utils.flatten_dict(log_assets)
            learning_rate = optimizer.param_groups[0]["lr"]
            log_assets.update({"lr": learning_rate})
            wandb.log(log_assets)
            log_assets.update({"epoch": epoch})
            pbar.set_postfix(log_assets)

    train_assets = TrainAssets(
        loss=average_meters["loss"].avg,
        cls_acc=average_meters["cls_acc"].avg if aux_params is not None else None,
        duration=int(time.time() - start),
    )
    return train_assets


class ValidAssets(NamedTuple):
    loss: float
    dice: float
    global_dice: float
    duration: int


def make_tta_model(model: nn.Module) -> nn.Module:
    transform = tta.Compose(
        [
            tta.HorizontalFlip(),
            # tta.VerticalFlip(),
            tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )
    tta_model = tta.SegmentationTTAWrapper(
        model, transform, merge_mode="mean", output_mask_key="preds"
    )
    return tta_model


TTAModelFn: TypeAlias = Callable[[nn.Module], nn.Module]


def valid_one_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    valid_loader: DataLoader,
    loss_fn: LossFn,
    device: torch.device,
    metrics_fn: MetricsFn,
    use_amp: bool = False,
    aux_params: AuxParams | None = None,
    cls_loss_fn: LossFn | None = None,
    debug: bool = False,
    metric_names: list[str] = ["loss", "dice"],
    forward_fn: ForwardFn = default_segmentation_forward_fn,
    postprocess_fn: PostProcessFn = default_postprocess_fn,
) -> ValidAssets:
    """Validate one epoch

    Args:
        fold (int): fold number

        epoch (int): epoch number

        model (nn.Module): model

        valid_loader (DataLoader): valid loader

        loss_fn (LossFn): loss function

        device (torch.device): device

        metrics_fn (MetricsFn): metrics function. Signature: (preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor

        use_amp (bool, optional): use amp. Defaults to False.

        aux_params (AuxParams, optional): aux params. Defaults to None.

        cls_loss_fn (LossFn, optional): cls loss function. Defaults to None. Signature: (torch.Tensor, torch.Tensor) -> torch.Tensor

        debug (bool, optional): debug mode. Defaults to False.

        metric_names (list[str], optional): metric names. Defaults to ["loss", "dice", "bce"].

        forward_fn (ForwardFn, optional): forward function. Defaults to default_segmentation_forward_fn. Signature: (model: nn.Module, ) -> torch.Tensor

        postprocess_fn (PostProcessFn, optional): postprocess function. Defaults to default_postprocess_fn. Signature: (torch.Tensor, torch.Tensor) -> torch.Tensor

        make_tta_model_fn (TTAModelFn, optional): make tta model function. Defaults to None. Signature: (nn.Module) -> nn.Module

    """
    model.eval()
    if "loss" not in metric_names:
        metric_names.append("loss")
    average_meters = init_average_meters(metric_names=metric_names)

    start = time.time()

    model = make_tta_model(model)

    # for global dice
    tp = 0
    positive_preds = 0
    positive_targets = 0

    pbar = tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        smoothing=0,
        dynamic_ncols=True,
        desc="Valid Per Epoch",
    )
    for step, batch in pbar:
        if isinstance(batch, tuple):
            batch = {"image": batch[0], "target": batch[1]}

        batch = send_tensor_to_device(batch=batch, device=device)
        batch_size = batch["target"].size(0)

        # --- Forward
        image = batch["image"]
        target = batch["target"]
        with (
            torch.inference_mode(),
            autocast(device_type=device.type, enabled=use_amp),
        ):
            # output = forward_fn(model=model, batch=batch)
            outs = model(image)
        # logits = outs["logits"].squeeze(1)
        preds = outs["preds"].squeeze(1)
        if target.shape[1:] != (256, 256):
            resized_target = F.interpolate(
                target.unsqueeze(1).float(), size=(256, 256), mode="bilinear"
            ).squeeze(1)
        else:
            resized_target = target.squeeze(1).float()

        output = ForwardOutputs(preds=preds, targets=resized_target)

        loss = loss_fn(output.preds, output.targets)

        # --- For Aux Heads
        # cls: (N, 1)
        if (
            aux_params is not None
            and output.cls_preds is not None
            and cls_loss_fn is not None
        ):
            cls_preds = output.cls_preds
            cls_target = _make_cls_label(output.targets)
            cls_target = cls_target.to(device, non_blocking=True)
            cls_loss = cls_loss_fn(cls_preds, cls_target)
            cls_loss = aux_params.cls_weight * cls_loss
            loss += cls_loss
            cls_bce = F.binary_cross_entropy_with_logits(cls_preds, cls_target)
            cls_accs = ((cls_preds > 0.5) == cls_target).sum() / batch_size
            wandb.log(
                {
                    f"valid/fold{fold}_cls_loss": cls_loss.item(),
                    f"valid/fold{fold}_cls_acc": cls_accs.item(),
                    f"valid/fold{fold}_cls_bce": cls_bce.item(),
                }
            )

        preds = output.preds.sigmoid().cpu().detach().numpy()
        targets = output.targets.cpu().detach().numpy()
        valid_metrics = metrics_fn(preds=preds, target=targets)

        # Comupte global dice
        tp += (preds * targets).sum()
        positive_preds += preds.sum()
        positive_targets += targets.sum()

        valid_metrics.update({"loss": loss.item()})

        # Aggregate metrics
        if not my_utils.is_nan(loss.item()):
            average_meters["loss"].update(value=loss.item(), n=batch_size)

        # Logging metrics
        for name, metric_value in valid_metrics.items():
            average_meters[name].update(value=metric_value, n=batch_size)

        valid_log_assets: dict[str, int | float] = {"epoch": epoch}
        for name, average_meter in average_meters.items():
            valid_log_assets[f"valid/fold{fold}_{name}_avg"] = average_meter.avg
            # valid_log_assets[f"valid/fold{fold}_{name}_std"] = average_meter.std

        wandb.log(valid_log_assets)
        pbar.set_postfix(valid_log_assets)

        if debug:
            nd_imgs = batch["image"].cpu().detach().permute(0, 2, 3, 1).numpy()
            nd_masks = batch["target"].cpu().detach().numpy()
            nd_preds = output.preds.sigmoid().cpu().detach().numpy()
            for i in range(batch_size):
                nd_img = nd_imgs[i]
                nd_mask = nd_masks[i]
                fig, ax = my_utils.plot_a_label_on_a_image(nd_img, nd_mask)
                ax.imshow(nd_preds, alpha=0.5, cmap="Reds", label="preds")
                ax.legend()
                fig.savefig(f"./debug/step{step}_{i}_mask.png")

    global_dice = 2 * tp / (positive_preds + positive_targets + 1e-6)
    wandb.log({f"valid/fold{fold}_global_dice": global_dice})
    logger.info(f"global dice: {global_dice}")

    valid_assets = ValidAssets(
        loss=average_meters["loss"].avg,
        dice=average_meters["dice"].avg,
        global_dice=global_dice,
        duration=int(time.time() - start),
    )
    return valid_assets
