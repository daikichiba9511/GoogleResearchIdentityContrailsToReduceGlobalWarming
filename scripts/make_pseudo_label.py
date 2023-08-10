import multiprocessing as mp
import uuid
from pathlib import Path
from typing import Callable, Sequence

import albumentations as A
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import typer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.factory import Config, init_config
from src.dataset import SegDataset
from src.models import ContrailsModel
from src.utils import (
    add_file_handler,
    filter_tiny_objects,
    get_called_time,
    get_stream_logger,
)

logger = get_stream_logger(20)

EXECUTE_TIME = get_called_time()


def get_loader(
    data_root_path: Path,
    test_aug_list: Sequence[Callable],
    batch_size: int,
    debug: bool = False,
) -> DataLoader:
    train_img_dirs = list((data_root_path / "train").glob("*"))
    valid_img_dirs = list((data_root_path / "validation").glob("*"))

    img_dirs = train_img_dirs + valid_img_dirs
    if debug:
        logger.info("debug mode: sample size is 50")
        img_dirs = img_dirs[:50]

    cls_df = pd.read_csv("./output/exp012/cls_infer_3d97fd13.csv")
    total_num = cls_df.shape[0]
    cls_df = cls_df[cls_df["pred"] < 0.9]
    skip_img_ids = cls_df["img_ids"].to_numpy()
    logger.info(f"the number of skip images: {len(skip_img_ids)} / {total_num}")

    # for i in range(5, 10):
    #     print(f"thr={i/10} {cls_df[cls_df['pred'] > (i/10)].shape[0] / len(cls_df) * 100}")
    # OUT:
    # thr=0.5 44.65181659394724
    # thr=0.6 44.353284204226775
    # thr=0.7 44.018566070297766
    # thr=0.8 43.62121876370013
    # thr=0.9 43.00466935276229

    test_aug = A.Compose(test_aug_list)  # type: ignore
    seg_dataset = SegDataset(
        img_dirs=img_dirs, transform_fn=test_aug, skip_image_ids=skip_img_ids
    )

    seg_dataloader = DataLoader(
        seg_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=mp.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )
    return seg_dataloader


class EnsembleModel:
    def __init__(
        self, models: Sequence[torch.nn.Module], on_cuda: bool = False
    ) -> None:
        self.models = models
        if on_cuda:
            self.models = [model.cuda() for model in self.models]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []  # shape: (n_models, batch_size, H, W)
        for model in self.models:
            output = model(x)
            pred = output["preds"].sigmoid()
            outputs.append(pred)
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.mean(dim=0)
        return outputs


def build_model(on_cuda: bool = False) -> EnsembleModel:
    # -- exp015: efficientnet-b8, img_size512
    config = init_config(Config, "configs.exp015")
    contrail_effb8 = ContrailsModel(
        encoder_name=config.encoder_name,
        encoder_weight=None,
        arch=config.arch,
        aux_params=config.aux_params,
    )
    ckpt = Path("./output/exp015/exp015-UNet-timm-efficientnet-b8-fold0.pth")
    logger.info(f"load checkpoint: {ckpt}")
    contrail_effb8.load_state_dict(torch.load(ckpt))
    contrail_effb8.eval()

    # -- exp018: mit_b5, img_size512
    config = init_config(Config, "configs.exp018")
    contrail_mitb5 = ContrailsModel(
        encoder_name=config.encoder_name,
        encoder_weight=None,
        arch=config.arch,
        aux_params=config.aux_params,
    )
    ckpt = Path("./output/exp018/exp018-UNet-mit_b5-fold0.pth")
    logger.info(f"load checkpoint: {ckpt}")
    contrail_mitb5.load_state_dict(torch.load(ckpt))
    contrail_mitb5.eval()

    model = EnsembleModel(models=[contrail_effb8, contrail_mitb5], on_cuda=on_cuda)
    return model


def _test_model():
    import time

    model = build_model()
    img = torch.randn(4, 3, 512, 512)
    start = time.time()
    with torch.inference_mode():
        pred = model(img)
    duration = time.time() - start
    logger.info(f"duration: {duration} with img shape {img.shape}")
    assert pred.shape == (4, 256, 256), f"pred.shape: {pred.shape}"

    logger.info("test on cuda")
    model = build_model(on_cuda=True)
    img = torch.randn(4, 3, 512, 512).cuda()
    start = time.time()
    with torch.inference_mode():
        pred = model(img)
    torch.cuda.synchronize()
    duration = time.time() - start
    logger.info(f"duration: {duration} with img shape {img.shape}")
    assert pred.shape == (4, 256, 256), f"pred.shape: {pred.shape}"


# TODO: 画像のデータの持ち方をどうにかする
def main(config_ver: str, debug: bool = False) -> None:
    config = init_config(Config, f"configs.{config_ver}")
    config.output_dir.mkdir(exist_ok=True, parents=True)
    execute_id = uuid.uuid4().hex[:8]
    add_file_handler(
        logger, filename=str(config.output_dir / f"cls_infer_{execute_id}.log")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    save_dir = Path(f"./input/imgs_with_pseudo_labels_{EXECUTE_TIME}")
    save_dir.mkdir(exist_ok=True, parents=True)

    loader = get_loader(
        data_root_path=config.data_root_path,
        test_aug_list=config.test_aug_list,
        batch_size=config.test_batch_size,
        debug=debug,
    )
    model = build_model(on_cuda=device.type == "cuda")

    preds = []
    with tqdm(
        loader,
        desc="Seg Infer",
        total=len(loader),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for idx, batch in enumerate(pbar):
            # batch shape:
            #   imgs: (batch_size, c, h, w)
            #   img_ids: (batch_size, )

            # print(batch)
            # import pdb
            #
            # pdb.set_trace()
            # break

            imgs, img_ids = batch
            batch_size = len(imgs)
            imgs = imgs.to(device, non_blocking=True)

            # outputs: (batch_size, h, w)
            with torch.inference_mode():
                outputs = model(imgs)
            outputs = outputs.detach().cpu()

            # TODO: どうやって一つのファイルにimgsとpredsを保存するか
            # メモリで持てるなら後でまとめて保存する
            # 無理そうならここで保存する
            # thrもimg_size=256に対して分析して決める
            imgs = F.interpolate(
                imgs.detach().cpu(),
                size=(256, 256),
                mode="bilinear",
                align_corners=False,
            )
            for batch_idx in range(batch_size):
                pred = outputs[batch_idx]
                pred_contrails = (pred > config.threshold).numpy().astype(np.uint8)
                pred_contrails = filter_tiny_objects(pred_contrails, thr=30)
                img = imgs[batch_idx].permute(1, 2, 0).numpy()
                img_id = img_ids[batch_idx]

                img_with_pred = np.dstack([img, pred_contrails]).astype(np.float16)

                if idx % 100 == 0:
                    logger.info(f"{img_id} {idx}/{len(preds)}")
                np.save(save_dir / f"{img_id}.npy", img_with_pred)

                # preds.append(
                #     {
                #         "pred": pred,
                #         "img_ids": img_id,
                #         "img_with_pred": img_with_pred,
                #     }
                # )

    # for idx, (pred, img_id, img_with_pred) in enumerate(preds):
    #     if idx % 100 == 0:
    #         logger.info(f"{img_id} {idx}/{len(preds)}")
    #     np.save(save_dir / f"{img_id}.npy", img_with_pred)


if __name__ == "__main__":
    """
    Usage:
        ```bash
        python scripts/infer_cls.py --config configs/exp009_2.py
        ```
    """
    typer.run(main)
    # _test_model()
