from enum import Enum
from typing import Any, Callable, Literal, TypeAlias

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from typing_extensions import assert_never

__all__ = [
    "LossTypeStr",
    "LossType",
    "LossFn",
    "get_loss",
]

LossTypeStr: TypeAlias = Literal["bce", "soft_bce", "dice", "dice_global", "bce_dice"]
LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossType(str, Enum):
    BCE = "bce"
    SoftBCE = "soft_bce"
    Dice = "dice"
    DiceGlobal = "dice_global"
    BCEDice = "bce_dice"


class DiceGlobalLoss(_Loss):
    def __init__(self, smooth: float = 1e-3):
        super().__init__()
        self.smooth = smooth

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice


class BCEDiceLoss(_Loss):
    def __init__(
        self,
        bce_weight: float | None = None,
        bce_smooth: float | None = None,
        dice_smooth: float = 0.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self._bce = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=bce_smooth)
        self._dice = smp.losses.DiceLoss(mode="binary", smooth=dice_smooth)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.bce_weight is None:
            return self._bce(pred, target) + self._dice(pred, target)

        return self.bce_weight * self._bce(pred, target) + (
            1 - self.bce_weight
        ) * self._dice(pred, target)


def get_loss(
    loss_type: LossTypeStr | LossType,
    loss_params: dict[str, Any] | None = None,
) -> LossFn:
    loss_type = LossType(loss_type)
    match loss_type:
        case LossType.BCE:
            if loss_params is None:
                return nn.BCEWithLogitsLoss()
            return nn.BCEWithLogitsLoss(**loss_params)
        case LossType.SoftBCE:
            if loss_params is None:
                return smp.losses.SoftBCEWithLogitsLoss()
            return smp.losses.SoftBCEWithLogitsLoss(**loss_params)
        case LossType.Dice:
            if loss_params is None:
                return smp.losses.DiceLoss()
            return smp.losses.DiceLoss(**loss_params)
        case LossType.DiceGlobal:
            if loss_params is None:
                return DiceGlobalLoss()
            return DiceGlobalLoss(**loss_params)
        case LossType.BCEDice:
            if loss_params is None:
                return BCEDiceLoss()
            return BCEDiceLoss(**loss_params)
        case _:
            assert_never(loss_type)


def _test_get_loss() -> None:
    loss = get_loss("bce", loss_params=None)
    print(loss)

    loss = get_loss(LossType.BCE, loss_params=None)
    print(loss)

    loss = get_loss(LossType.BCE, loss_params={"weight": torch.tensor([1, 2])})
    print(loss)


if __name__ == "__main__":
    _test_get_loss()
