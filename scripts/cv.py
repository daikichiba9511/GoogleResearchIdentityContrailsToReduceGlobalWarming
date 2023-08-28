import multiprocessing as mp
from pathlib import Path

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from configs.factory import made_config
from scripts.train_seg2 import ContrailDatasetV3
from src.metrics import GlobalDice, MetricsFns
from src.models import ContrailsModel, CustomedUnet, builded_model
from src.train_tools import AverageMeter, seed_everything
from src.utils import (
    add_file_handler,
    get_stream_logger,
    plot_for_debug,
    plot_preds_with_label_on_image,
)

logger = get_stream_logger(20)


class EnsembleModel(nn.Module):
    def __init__(
        self,
        models: list[ContrailsModel | CustomedUnet],
        weights: list[float] | None = None,
    ):
        super().__init__()
        self._models = models
        if weights is None or len(weights) == 0:
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
    models = []
    weights = []

    # config 37_1
    _model = builded_model(
        made_config("unet_v1", debug=False), disable_compile=False, fold=0
    )
    _model.load_state_dict(torch.load("./output/"))
    models.append(_model)
    weights.append(0.5)

    assert len(models) == len(weights)
    ensemble_model = EnsembleModel(models=models, weights=weights)
    return ensemble_model


def main(debug: bool = False, batch_size: int = 32) -> None:
    description = ""
    logging_fp = f"./cv_logs/cv_{description}.log"
    Path(logging_fp).parent.mkdir(exist_ok=True, parents=True)
    if Path(logging_fp).exists():
        raise FileExistsError(f"{logging_fp} already exists.")

    seed_everything(42)
    add_file_handler(logger, logging_fp)

    image_paths = list(
        Path(
            "./input/google-research-identify-contrails-reduce-global-warming/validation/"
        ).glob("*")
    )
    if debug:
        image_paths = image_paths[:100]
    record_ids = [p.stem for p in image_paths]
    aug_fn = A.Compose([ToTensorV2()])
    valid_dataset = ContrailDatasetV3(
        image_paths, record_ids, "valid", aug_fn, fix_offset=True
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
    dices = AverageMeter("dice")
    accs = AverageMeter("accs")
    global_dice = GlobalDice()

    for i, batch in enumerate(data_loader):
        image, mask = batch["image"], batch["mask"]
        with torch.inference_mode():
            preds = ensemble_model(image)

        preds = preds.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        # Metrics
        metrics = MetricsFns.calc_metrics(preds, mask)
        dices.update(metrics["dice"])
        accs.update(metrics["accs"])
        global_dice.update(preds, mask)

        # Visualize
        if i % 10 == 0:
            record_ids = batch["record_id"]
            for j in range(len(preds)):
                mask_i = mask[j]
                if mask_i.flatten().sum() == 0:
                    continue
                plot_for_debug(
                    image[j],
                    mask[j],
                    preds[j],
                    Path(f"./cv_logs/{description}/visualize"),
                    record_ids[j],
                )
                fig, ax = plot_preds_with_label_on_image(
                    image[j],
                    preds[j],
                    mask[j],
                )
                fig.savefig(f"./cv_logs/{description}/visualize/batch{i}_{j}.png")

    logger.info(
        f"""
    Metrics
    ---------------------------

    model_num   : {len(ensemble_model._models)}

    global dice : {global_dice.value}
    dice        : {dices.avg}
    acc         : {accs.avg}
    """
    )
    logger.info(" === cv finished === ")


if __name__ == "__main__":
    main()
