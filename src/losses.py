from enum import Enum
from typing import Callable, Literal, TypeAlias

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from typing_extensions import assert_never

__all__ = [
    "LossTypeStr",
    "LossType",
    "LossFn",
    "get_loss",
]

LossTypeStr: TypeAlias = Literal["bce", "soft_bce"]
LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossType(str, Enum):
    BCE = "bce"
    SoftBCE = "soft_bce"


def get_loss(
    loss_type: LossTypeStr | LossType,
    loss_params: dict[str, object] | None = None,
) -> LossFn:
    loss_type = LossType(loss_type)
    match loss_type:
        case LossType.BCE:
            if loss_params is not None:
                loss = nn.BCEWithLogitsLoss(**loss_params)
                return loss
            else:
                loss = nn.BCEWithLogitsLoss()
                return loss
        case LossType.SoftBCE:
            if loss_params is not None:
                loss = smp.losses.SoftBCEWithLogitsLoss(**loss_params)
                return loss
            else:
                loss = smp.losses.SoftBCEWithLogitsLoss()
                return loss
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
