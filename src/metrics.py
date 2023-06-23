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


def calc_metrics(preds: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    dice_coef = dice(preds, target)

    metrics = {
        "dice": dice_coef,
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
