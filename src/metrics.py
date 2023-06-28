import numpy as np
import torch

__all__ = ["dice", "calc_metrics"]


def dice(pred: np.ndarray, mask: np.ndarray) -> float:
    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    dice = (2.0 * intersection) / (pred_sum + mask_sum)
    dice = np.mean(dice)
    return dice.item()


def dice_coef(
    preds: np.ndarray, target: np.ndarray, thr: float = 0.5, eps: float = 1e-7
) -> float:
    target = target.astype(np.float32).flatten()
    preds = (preds > thr).astype(np.float32).flatten()
    intersection = np.sum(preds * target)
    union = np.sum(preds) + np.sum(target)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice


def calc_metrics(preds: np.ndarray, target: np.ndarray) -> dict[str, float]:
    dice_coef_value = dice(preds, target)
    # dice_coef_value = dice_coef(preds, target, thr=0.5, eps=1e-7)

    metrics = {
        "dice": dice_coef_value,
    }
    return metrics


def _test_dice() -> None:
    pred = torch.randint(0, 2, size=(256, 256))
    mask = torch.randint(0, 2, size=(256, 256))
    dice_score = dice(pred, mask)
    print(dice_score)
    print("Run test_dice() successfully")


if __name__ == "__main__":
    _test_dice()
