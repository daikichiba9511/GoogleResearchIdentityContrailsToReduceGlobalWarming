import multiprocessing as mp
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from src.dataset import ContrailsDatasetV2
from src.metrics import calc_metrics
from src.models import ContrailsModel, CustomedUnet
from src.train_tools import init_average_meters, seed_everything
from src.utils import add_file_handler, get_stream_logger

logger = get_stream_logger(20)


class EnsembleModel(nn.Module):
    def __init__(
        self, models: list[ContrailsModel], weights: list[float] | None = None
    ):
        super().__init__()
        self._models = models
        if weights is None:
            self._weights = [1.0 / len(models)] * len(models)
        else:
            self._weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preds = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]))
        for i, model in enumerate(self._models):
            outs = model(x)
            preds[i] = self._weights[i] * outs["preds"]
        return preds


def build_ensemble_model() -> EnsembleModel:
    ...


def main(debug: bool = False, batch_size: int = 32) -> None:
    description = ""
    logging_fp = f"./cv_logs/cv_{description}.log"
    Path(logging_fp).parent.mkdir(exist_ok=True, parents=True)
    if Path(logging_fp).exists():
        raise FileExistsError(f"{logging_fp} already exists.")

    seed_everything(42)
    add_file_handler(logger, logging_fp)

    df = pd.read_csv("./input/prepared_np_imgs_weight1/metadata.csv")
    valid_df = df.query("not is_train").reset_index(drop=True)
    if debug:
        valid_df = valid_df[:100]
    image_paths = valid_df["image_path"].to_numpy().tolist()
    mask_paths = valid_df["mask_path"].to_numpy().tolist()
    aug_fn = A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2()])
    valid_dataset = ContrailsDatasetV2(
        img_paths=image_paths,
        transform_fn=aug_fn,
        phase="val",
        mask_paths=mask_paths,
    )
    data_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    ensemble_model = build_ensemble_model()
    average_meters = init_average_meters(["dice"])

    tp = 0
    pred_sum = 0
    mask_sum = 0
    for i, batch in enumerate(data_loader):
        image, mask = batch["image"], batch["mask"]
        with torch.inference_mode():
            preds = ensemble_model(image)

        preds = preds.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        metrics = calc_metrics(preds, mask)
        for k, v in metrics.items():
            average_meters[k].update(v, batch_size)

        tp += (preds * mask).sum()
        pred_sum += preds.sum()
        mask_sum += mask.sum()

    global_dice = (2.0 * tp + 1e-7) / (pred_sum + mask_sum + 1e-7)

    report = f"""
    Global dice: {global_dice}

    dice: {average_meters['dice'].avg}

    """
    logger.info(report)
    logger.info(" === cv finished === ")


if __name__ == "__main__":
    main()
