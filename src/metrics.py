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


def _test_dice() -> None:
    pred = torch.randn((3, 1, 256, 256)).numpy()
    mask = torch.randint(0, 2, size=(3, 1, 256, 256)).numpy()
    print(pred.shape, mask.shape)
    metrics = calc_metrics(pred, mask)
    print(metrics)
    print("Run test_dice() successfully")


if __name__ == "__main__":
    _test_dice()
