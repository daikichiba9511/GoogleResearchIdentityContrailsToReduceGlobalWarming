import numpy as np
import segmentation_models_pytorch as smp
import torch

__all__ = ["calc_metrics"]


def _dice(pred: np.ndarray, mask: np.ndarray, eps: float = 1e-7) -> float:
    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    dice = (2.0 * intersection) / (pred_sum + mask_sum + eps)
    dice = np.mean(dice)
    return dice.item()


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
    if not isinstance(preds, torch.Tensor):
        preds = torch.from_numpy(preds).float()
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target)

    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    dice_coef_value = dice(preds, target)
    # dices = {}
    # for thr in np.arange(0.3, 0.85, 0.5):
    #     dice_coef_value = dice_coef(preds, target, thr=thr, eps=1e-5)
    #     dices[f"dice_{thr}"] = dice_coef_value

    metrics = {
        "dice": 1 - dice_coef_value.item(),
        # **dices,
    }
    return metrics


def _test_dice() -> None:
    pred = torch.randn((3, 1, 256, 256)).numpy()
    mask = torch.randint(0, 2, size=(3, 1, 256, 256)).numpy()
    print(pred.shape, mask.shape)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    metrics = calc_metrics(pred, mask)
    print(metrics)
    print("Run test_dice() successfully")


if __name__ == "__main__":
    _test_dice()
