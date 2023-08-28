from enum import Enum
from typing import Any, Callable, Literal, TypeAlias

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from typing_extensions import assert_never

__all__ = [
    "LossTypeStr",
    "LossType",
    "LossFn",
    "get_loss",
]

LossTypeStr: TypeAlias = Literal[
    "bce", "soft_bce", "dice", "dice_global", "bce_dice", "srloss"
]
LossFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class LossType(str, Enum):
    BCE = "bce"
    SoftBCE = "soft_bce"
    Dice = "dice"
    DiceGlobal = "dice_global"
    BCEDice = "bce_dice"
    SRLoss = "srloss"


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


# SRLoss
def hough_transform(batch, threshold=50, return_coordinates=False):
    """
    Reference:
    [1] https://github.com/junzis/contrail-net/blob/main/loss.py#L123
    """
    height, width = batch[0].squeeze().shape

    thetas = torch.arange(0, 180, 0.5)
    d = torch.sqrt(torch.tensor(width) ** 2 + torch.tensor(height) ** 2)
    rhos = torch.arange(-d, d, 3)

    cos_thetas = torch.cos(torch.deg2rad(thetas))
    sin_thetas = torch.sin(torch.deg2rad(thetas))

    hough_matrices = torch.Tensor(
        batch.shape[0], rhos.shape[0] - 1, thetas.shape[0] - 1
    )

    for i, img in enumerate(batch):
        img = img.squeeze()
        points = torch.argwhere(img > 0.5).type_as(cos_thetas)
        rho_values = torch.matmul(points, torch.stack((sin_thetas, cos_thetas)))

        accumulator, (theta_vals, rho_vals) = torch.histogramdd(
            torch.stack(
                (
                    torch.tile(thetas, (rho_values.shape[0],)),
                    rho_values.ravel(),
                )
            ).T,
            bins=[thetas, rhos],
        )

        accumulator = torch.transpose(accumulator, 0, 1)

        if return_coordinates:
            hough_lines = torch.argwhere(accumulator > threshold)
            rho_idxs, theta_idxs = hough_lines[:, 0], hough_lines[:, 1]
            hough_rhos, hough_thetas = rhos[rho_idxs], thetas[theta_idxs]
            hough_coordinates = torch.stack((hough_rhos, hough_thetas))
            return hough_coordinates
        else:
            hough_matrix = torch.where(accumulator > threshold, 1, 0)
            hough_matrices[i] = hough_matrix
            return hough_matrices


class SRLoss(_Loss):
    """
    Reference:
    [1] https://github.com/junzis/contrail-net/blob/main/loss.py#L166
    """

    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
        dh_weight=0.5,
    ):
        """Dice loss for image segmentation task with Hough Transform constraint."""
        super(SRLoss, self).__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.dh_weight = dh_weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        # bs = y_true.size(0)
        # dims = (1, 2, 3)

        # y_true = y_true.view(bs, 1, -1)
        # y_pred = y_pred.view(bs, 1, -1)

        predict, target = y_pred, y_true.type_as(y_pred)

        smooth = self.smooth
        eps = self.eps

        intersection = torch.sum(predict * target)
        cardinality = torch.sum(predict + target)
        dice = (2 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
        if self.log_loss:
            dice_loss = -torch.log(dice.clamp_min(self.eps))
        else:
            dice_loss = 1 - dice

        # compute customized hough loss
        hough_predict = hough_transform(predict)
        hough_target = hough_transform(target)

        h_intersection = torch.sum(hough_predict * hough_target)
        h_cardinality = torch.sum(hough_predict + hough_target)
        dice = (2 * h_intersection + smooth) / (h_cardinality + smooth).clamp_min(eps)

        if self.log_loss:
            hough_dice_loss = -torch.log(dice.clamp_min(self.eps))
        else:
            hough_dice_loss = 1 - dice

        loss = self.dh_weight * dice_loss + (1 - self.dh_weight) * hough_dice_loss

        return loss


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
            return smp.losses.SoftBCEWithLogitsLoss(**loss_params, reduction="none")
        case LossType.Dice:
            if loss_params is None:
                return smp.losses.DiceLoss(mode="binary")
            return smp.losses.DiceLoss(**loss_params)
        case LossType.DiceGlobal:
            if loss_params is None:
                return DiceGlobalLoss()
            return DiceGlobalLoss(**loss_params)
        case LossType.BCEDice:
            if loss_params is None:
                return BCEDiceLoss()
            return BCEDiceLoss(**loss_params)
        case LossType.SRLoss:
            if loss_params is None:
                return SRLoss()
            return SRLoss(**loss_params)
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
