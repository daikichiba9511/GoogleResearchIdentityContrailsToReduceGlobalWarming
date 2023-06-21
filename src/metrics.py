import torch

__all__ = ["dice", "calc_metrics"]


def dice(pred: torch.Tensor, mask: torch.Tensor) -> float:
    intersection = (pred * mask).sum()
    pred_sum = pred.sum()
    mask_sum = mask.sum()
    dice = (2.0 * intersection) / (pred_sum + mask_sum)
    dice = torch.mean(dice)
    return dice.item()


def calc_metrics(pred: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    dice_coef = dice(pred, mask)

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
