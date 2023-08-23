import numpy as np
import segmentation_models_pytorch as smp
import torch

__all__ = ["calc_metrics"]


def _dice(pred: np.ndarray, mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(np.float32).flatten()
    mask = mask.astype(np.float32).flatten()
    intersection = (pred * mask).sum()
    union = pred.sum() + mask.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def dice_coef(
    preds: np.ndarray, target: np.ndarray, thr: float = 0.5, eps: float = 1e-7
) -> float:
    target = target.astype(np.float32).flatten()
    preds = (preds > thr).astype(np.float32).flatten()
    intersection = np.sum(preds * target)
    union = preds.sum() + target.sum()
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def calc_metrics(
    preds: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor
) -> dict[str, float]:
    # if not isinstance(preds, torch.Tensor):
    #     preds = torch.from_numpy(preds).float()
    # if not isinstance(target, torch.Tensor):
    #     target = torch.from_numpy(target).float()

    # dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    # dice_coef_value = dice(preds, target).item()

    dice_coef_value = _dice(preds, target)  # type: ignore

    metrics = {
        "dice": dice_coef_value,
        # **dices,
    }
    return metrics


class GlobalDice:
    def __init__(self, eps: float = 1e-8) -> None:
        self._tp = 0
        self._preds_sum = 0
        self._mask_sum = 0
        self._eps = eps

    def update(self, preds: torch.Tensor, mask: torch.Tensor) -> None:
        self._tp += (preds * mask).sum().item()
        self._preds_sum += preds.sum().item()
        self._mask_sum += mask.sum().item()

    @property
    def value(self) -> float:
        """Return global dice score

        NOTE:
            2 * Intersection / Union
            = 2 * (TP + eps) / (preds_sum + mask_sum + eps)

            where:
                - TP: True Positive

        Returns:
            float: global dice score
        """
        return (2 * self._tp + self._eps) / (
            self._preds_sum + self._mask_sum + self._eps
        )


class MetricsFns:
    @staticmethod
    def dice(preds: torch.Tensor, mask: torch.Tensor, eps: float = 1e-4) -> float:
        flatten_preds = preds.view(-1)
        flatten_mask = mask.view(-1)
        intersection = (flatten_preds * flatten_mask).sum()
        union = flatten_preds.sum() + flatten_mask.sum()
        dice_value = (2 * intersection + eps) / (union + eps)
        return dice_value.item()

    @classmethod
    def calc_metrics(cls, preds: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        dice = cls.dice(preds, mask)
        accs = (preds == mask).float().mean().item()
        return {"dice": dice, "accs": accs}


def _test_dice() -> None:
    pred = torch.randn((3, 1, 256, 256)).numpy()
    mask = torch.randint(0, 2, size=(3, 1, 256, 256)).numpy()
    print(pred.shape, mask.shape)
    metrics = calc_metrics(pred, mask)
    print(metrics)
    print("Run test_dice() successfully")


if __name__ == "__main__":
    _test_dice()
