"""読み込んだ画像にラベルを含んでいるかを推論するスクリプト
"""
import multiprocessing as mp
import uuid
from pathlib import Path
from typing import Callable, Sequence

import albumentations as A
import pandas as pd
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
    config.output_dir.mkdir(exist_ok=True, parents=True)
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

    preds = []
    with tqdm(
        loader,
        desc="Seg Infer",
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for idx, batch in enumerate(pbar):
            # batch shape: (batch_size, mini_batch_size|img_id), mini_batch: (mini_batch_size, c, h, w)
            # print(batch)
            # pdb.set_trace()

            imgs, img_ids = batch
            reconstrcut_imgs, reconstrcut_img_ids = [], []
            batch_size = len(imgs)
            for batch_idx in range(batch_size):
                # shape: (7, h, w), excpet n_times_before=4
                imgs_have_same_ids = imgs[batch_idx]
                img_ids_have_same_ids = [img_id_t[batch_idx] for img_id_t in img_ids]
                reconstrcut_imgs.append(imgs_have_same_ids)
                reconstrcut_img_ids += img_ids_have_same_ids

            # shape: (batch_size * mini_batch_size, c, h, w)
            reconstruct_imgs = torch.cat(reconstrcut_imgs, dim=0)
            assert (
                reconstruct_imgs.ndim == 4
            ), f"reconstruct_imgs.shape: {reconstruct_imgs.shape}"

            batch_size = len(reconstruct_imgs)
            imgs = reconstruct_imgs.to(device, non_blocking=True)

            with torch.inference_mode():
                outputs = model(imgs)

            # shape: (batch_size, )
            cls_pred = outputs["cls_logits"].sigmoid().detach().cpu().numpy()
            assert len(cls_pred) == len(reconstrcut_img_ids)
            # print(batch_size, len(cls_pred), len(reconstrcut_img_ids))
            for batch_idx in range(batch_size):
                preds.append(
                    {
                        "pred": cls_pred[batch_idx],
                        "img_ids": reconstrcut_img_ids[batch_idx],
                    }
                )

    pred_df = pd.DataFrame(preds)
    print(pred_df)
    pred_df.to_csv(config.output_dir / f"cls_infer_{execute_id}.csv", index=False)


if __name__ == "__main__":
    """
    Usage:
        ```bash
        python scripts/infer_cls.py --config configs/exp009_2.py
        ```
    """
    typer.run(main)
