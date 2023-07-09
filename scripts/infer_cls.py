"""読み込んだ画像にラベルを含んでいるかを推論するスクリプト
"""
import multiprocessing as mp
import uuid
from pathlib import Path
from typing import Callable, Sequence

import albumentations as A
import timm
import torch
import typer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.factory import Config, init_config
from src.dataset import ClsDataset
from src.models import ContrailsModel
from src.utils import add_file_handler, get_stream_logger

logger = get_stream_logger(20)


def get_loader(
    data_root_path: Path,
    test_aug_list: Sequence[Callable],
    batch_size: int,
    debug: bool = False,
) -> DataLoader:
    train_img_dirs = list((data_root_path / "train").glob("*"))
    valid_img_dirs = list((data_root_path / "valid").glob("*"))

    img_dirs = train_img_dirs + valid_img_dirs
    if debug:
        logger.info("debug mode: sample size is 50")
        img_dirs = img_dirs[:50]

    test_aug = A.Compose(test_aug_list)
    cls_dataset = ClsDataset(img_dirs=img_dirs, transform_fn=test_aug)

    cls_dataloader = DataLoader(
        cls_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    return cls_dataloader


def build_model(config: Config) -> torch.nn.Module:
    model = ContrailsModel(
        encoder_name=config.encoder_name,
        encoder_weight=None,
        arch=config.arch,
        aux_params=config.aux_params,
    )
    ckpt = config.checkpoints[0]
    logger.info(f"load checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    return model


def main(config_ver: str, debug: bool = False) -> None:
    config = init_config(Config, f"configs.{config_ver}")
    execute_id = uuid.uuid4().hex[:8]
    add_file_handler(
        logger, filename=str(config.output_dir / f"cls_infer_{execute_id}.log")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = get_loader(
        data_root_path=config.data_root_path,
        test_aug_list=config.test_aug_list,
        batch_size=config.test_batch_size,
        debug=debug,
    )
    model = build_model(config)
    model = model.to(device, non_blocking=True)

    with tqdm(
        loader, desc="Cls Infer", total=len(loader), dynamic_ncols=True, leave=False
    ) as pbar:
        for idx, (imgs, img_ids) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)

            with torch.inference_mode():
                outputs = model(imgs)

            cls_pred = outputs["cls_logits"].sigmoid().detach().cpu().numpy()


if __name__ == "__main__":
    """
    Usage:
        ```bash
        python scripts/infer_cls.py --config configs/exp009_2.py
        ```
    """
    typer.run(main)
