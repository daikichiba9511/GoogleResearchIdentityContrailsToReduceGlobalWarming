import os
import random
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, Iterable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

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
        self.backup = {}
        self.backup_eps = {}
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
                self.scaler.scale(adv_loss).backward()
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
        return f"Metrics {self.name}: Avg {self.avg}, Row values {self.rows}"

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.rows: list[float | int] = []

    def update(self, value: float | int, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        self.rows.append(value)

    def to_dict(self) -> dict[str, list[float | int] | str | float]:
        return {"name": self.name, "avg": self.avg, "row_values": self.rows}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.autograd.set_detect_anomaly(False)


def _make_cls_label(mask: torch.Tensor) -> torch.Tensor:
    """make classification label from mask

    Args:
        mask (torch.Tensor): mask, (B, H, W)

    Returns:
        torch.Tensor: classification label
    """
    cls_label = mask.sum(dim=(1, 2)) > 0
    assert cls_label.shape == (mask.shape[0],)
    return cls_label


def _freeze_model(model: nn.Module, freeze_keys: Iterable[str] = ["encoder"]) -> None:
    """freeze model parameters specified by freeze_keys

    Args:
        model (nn.Module): model
    """
    for name, param in model.named_parameters():
        contains = [key in name for key in freeze_keys]
        if any(contains):
            param.requires_grad = False


def scheduler_step(scheduler: optim.lr_scheduler.LRScheduler, loss: float) -> None:
    if loss in [float("inf"), float("-inf"), None, torch.nan]:
        return

    scheduler.step(loss)


@dataclass(frozen=True)
class AWPParams:
    adv_lr: float
    adv_eps: float
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

    do_label_noise: bool
    label_noise_prob: float


@dataclass(frozen=True)
class AuxParams:
    cls_weight: float


@dataclass(frozen=True)
class FreezeParams:
    start_epoch_to_freeze_model: int
    freeze_keys: list[str]


def train_one_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    criterion: LossFn,
    scaler: GradScaler,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    max_grad_norm: float = 1000.0,
    criterion_cls: LossFn | None = None,
    schedule_per_step: bool = False,
    use_amp: bool = False,
    grad_accum_step_num: int = 1,
    awp_params: AWPParams | None = None,
    aug_params: AugParams | None = None,
    aux_params: AuxParams | None = None,
    freeze_params: FreezeParams | None = None,
) -> dict[str, object]:
    if awp_params is not None:
        awp = AWP(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            adv_lr=awp_params.adv_lr,
            adv_eps=awp_params.adv_eps,
            start_epoch=awp_params.start_epoch,
            adv_step=awp_params.adv_step,
        )

    # used for freeze model
    is_frozen = False

    running_losses = AverageMeter(name="train_loss")
    with tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        dynamic_ncols=True,
        desc="Train Per Epoch",
    ) as pbar:
        for step, (images, target) in pbar:
            model.train()

            if (
                aug_params is not None
                and aug_params.cutmix
                and np.random.rand() <= aug_params.cutmix_prob
            ):
                images, target, _, _ = mixup(
                    images, target, alpha=aug_params.mixup_alpha
                )

            if (
                aug_params is not None
                and aug_params.mixup
                and np.random.rand() <= aug_params.mixup_prob
            ):
                images, target, _, _ = cutmix(
                    images, target, alpha=aug_params.cutmix_alpha
                )

            if (
                aug_params is not None
                and aug_params.label_noise
                and np.random.rand() <= aug_params.label_noise_prob
            ):
                images, target, _ = label_noise(images, target)

            batch_size = target.size(0)
            images = images.contiguous().to(device, non_blocking=True)
            target = target.contiguous().to(device, non_blocking=True)

            if aux_params is not None:
                target_cls = _make_cls_label(target)

            if not is_frozen and epoch > freeze_params.start_epoch_to_freeze_model:
                logger.info(f"freeze model with {freeze_params.freeze_keys}")
                _freeze_model(model, freeze_keys=freeze_params.freeze_keys)
                is_frozen = True

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                logit = outputs["logit"]
                loss = criterion(logit, target)
                loss_mask = loss

                if (
                    aux_params is not None
                    and any(["cls" in out_key for out_key in outputs.keys()])
                    and criterion_cls is not None
                ):
                    cls_logits1 = outputs["cls_logits"]
                    loss_cls1 = aux_params.cls_weight * criterion_cls(
                        cls_logits1, target_cls
                    )
                    loss_cls = loss_cls1
                else:
                    loss_cls = 0

                loss = loss_mask + loss_cls
                loss /= grad_accum_step_num

            running_losses.update(value=loss.item(), n=batch_size)
            scaler.scale(loss).backward()

            if awp_params is not None and epoch >= awp_params.start_epoch:
                awp.attack_backward(images, target, epoch)

            if (step + 1) % grad_accum_step_num == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if schedule_per_step:
                    scheduler_step(scheduler, loss=loss)

                log_assets = {
                    "fold": f"{fold}",
                    "epoch": f"{epoch}",
                    "loss_avg": f"{running_losses.avg:.4f}",
                    "loss": f"{loss.item():.4f}",
                }

                learning_rate = optimizer.param_groups[0]["lr"]
                wandb_log_assets = {
                    f"fold{fold}_train_loss": loss.item(),
                    "learning_rate": learning_rate,
                }

                if aux_params is not None:
                    log_assets.update(
                        {
                            "cls_loss": f"{loss_cls.item():.4f}",
                        }
                    )
                    wandb_log_assets.update(
                        {
                            f"fold{fold}_cls_train_loss": loss_cls.item(),
                        }
                    )

                pbar.set_postfix(log_assets)
                wandb.log(wandb_log_assets)

    train_assets = {
        "train_loss": running_losses.avg,
    }
    return train_assets


class MetricsFn(Protocol):
    def __call__(self, preds: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
        ...


def valid_per_epoch(
    fold: int,
    epoch: int,
    model: nn.Module,
    valid_loader: DataLoader,
    criterion: LossFn,
    device: torch.device,
    use_amp: bool = False,
    criterion_cls: LossFn | None = None,
    metrics_fn: MetricsFn | None = None,
    log_prefix: str = "",
    aux_params: AuxParams | None = None,
) -> dict[str, object]:
    model.eval()
    valid_losses = AverageMeter(name="valid_loss")
    valid_bces = AverageMeter(name="valid_bce")

    with tqdm(
        enumerate(valid_loader),
        total=len(valid_loader),
        smoothing=0,
        dynamic_ncols=True,
        desc="Valid Per Epoch",
    ) as pbar:
        for step, (image, target) in pbar:
            batch_size = target.size(0)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with torch.inference_mode():
                with autocast(device_type=device.type, enabled=use_amp):
                    output = model(image)
                logits = output["logits"]
                loss_mask = criterion(logits, target)

                # cls: (N, 1)
                if (
                    aux_params is not None
                    and any(["cls" in out_key for out_key in output.keys()])
                    and criterion_cls is not None
                ):
                    target_cls = _make_cls_label(target).to(device, non_blocking=True)
                    cls_logits = output["cls_logits"]
                    loss_cls = criterion_cls(cls_logits, target_cls)
                    loss_cls = aux_params.weight_cls * loss_cls
                else:
                    loss_cls = 0

                loss = loss_mask + loss_cls

            valid_losses.update(value=loss.item(), n=batch_size)

            # make a whole image prediction
            # y_preds: (N, H, W)
            y_preds = torch.sigmoid(logits)
            y_preds = y_preds.to("cpu").detach().numpy()

            valid_metrics = metrics_fn(
                target=target.to("cpu").detach().numpy(), preds=y_preds
            )

            # assert y_preds.shape == (batch_size, 1, crop_size, crop_size)
            # start_idx = step * batch_size
            # end_idx = start_idx + batch_size
            # for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            #     mask_preds[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            #     mask_count[y1:y2, x1:x2] += np.ones((crop_size, crop_size))

            # TODO: どの指標を管理するか考える
            valid_log_assets = {
                f"{log_prefix}fold{fold}_valid_loss": loss.item(),
                **valid_metrics,
            }
            valid_wandb_log_assets = {
                "valid_loss": loss.item(),
                **valid_metrics,
            }
            valid_wandb_log_assets = {
                f"{log_prefix}fold{fold}_{metrics_name}": value
                for metrics_name, value in valid_wandb_log_assets.items()
            }

            if aux_params is not None:
                bce = F.binary_cross_entropy_with_logits(
                    input=cls_logits, target=target_cls
                )
                accs = ((cls_logits > 0.5) == target_cls).sum().item() / batch_size
                valid_bces.update(value=bce.item(), n=batch_size)

                valid_log_assets.update(
                    {
                        "cls_loss": f"{loss_cls.item():.4f}",
                        "avg_cls_loss": f"{loss_cls.item():.4f}",
                        "cls_acc": f"{accs:.4f}",
                    }
                )
                valid_wandb_log_assets.update(
                    {
                        f"fold{fold}_cls_valid_loss": loss_cls.item(),
                        f"fold{fold}_cls_acc": accs,
                    }
                )

            pbar.set_postfix(valid_log_assets)
            wandb.log(valid_wandb_log_assets)

    # logger.info(
    #     f"mask_count_min: {mask_count.min()}, mask_count_max: {mask_count.max()}, zero_sum: {(mask_count == 0).sum()}"
    # )
    # mask_preds /= mask_count + 1e-7
    valid_assets = {
        "valid_avg_loss": valid_losses.avg,
    }
    return valid_assets
