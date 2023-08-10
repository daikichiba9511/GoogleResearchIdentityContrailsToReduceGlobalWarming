"""読み込んだ画像にラベルを含んでいるかを推論するスクリプト
"""
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import typer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from src.dataset import get_false_color, read_record
from src.utils import (
    filter_tiny_objects,
    get_called_time,
    get_stream_logger,
    plot_a_label_on_a_image,
    plot_images_with_labels,
)

logger = get_stream_logger(20)

EXECUTE_TIME = get_called_time()


class PlotDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        image_ids: Sequence[str],
        image_size: int = 224,
        transform_fn: Callable | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.image_size = image_size
        self.transform_fn = transform_fn

    def __getitem__(self, index: int) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
        contrails_image_path = self.image_paths[index]
        image_id = self.image_ids[index]
        record_data = read_record(contrails_image_path)
        n_times_before = 4
        false_color_img = get_false_color(record_data)
        raw_image = false_color_img[..., n_times_before]
        raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)
        raw_individual_mask = np.load(
            contrails_image_path.parent / image_id / "human_individual_masks.npy"
        )
        raw_pixel_masks = np.load(
            contrails_image_path.parent / image_id / "human_pixel_masks.npy"
        )
        raw_pixel_masks = np.reshape(raw_pixel_masks, (256, 256)).astype(np.float32)

        return raw_image, image_id, raw_pixel_masks, raw_individual_mask

    def __len__(self) -> int:
        return len(self.image_paths)


# TODO: 画像のデータの持ち方をどうにかする
def main() -> None:
    output_dir = Path("./output/eda/labels_on_imgs")
    data_root_path = Path(
        "./input/google-research-identify-contrails-reduce-global-warming"
    )
    debug = True

    output_dir.mkdir(exist_ok=True, parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    train_img_dirs = list((data_root_path / "train").glob("*"))
    # valid_img_dirs = list((data_root_path / "validation").glob("*"))
    img_dirs = train_img_dirs
    if debug:
        logger.info("debug mode: sample size is 50")
        img_dirs = img_dirs[:50]

    seg_dataset = PlotDataset(
        image_paths=img_dirs,
        image_ids=[img_dir.name for img_dir in img_dirs],
    )
    pbar = tqdm(
        iter(seg_dataset),
        desc="Seg Infer",
        total=len(seg_dataset),
        dynamic_ncols=True,
        leave=False,
    )

    # NOTE:
    # 可視化したいものはなにか
    # - 1. 画像に上にラベルをつけたもの
    # - 3. その結果としてのラベル同士の比較
    for idx, (img, img_id, pixel_mask, individual_mask) in enumerate(pbar):
        if idx > 500:
            break

        if np.sum(pixel_mask) == 0 or np.any(np.sum(individual_mask, axis=(0, 1)) == 0):
            continue

        # - 1. 画像に上にラベルをつけたもの
        fig, ax = plot_a_label_on_a_image(image=img, label=pixel_mask)
        fig.savefig(output_dir / f"{img_id}_pixel_mask.png")

        # - 3. その結果としてのラベル同士の比較
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        assert isinstance(fig, plt.Figure) and isinstance(ax, plt.Axes)
        avg_mask = np.mean(individual_mask, axis=-1)
        ax.imshow(avg_mask, alpha=0.5)
        ax.imshow(pixel_mask, alpha=0.5)
        fig.savefig(output_dir / f"{img_id}_pixel_mask_and_avg_mask_comparison.png")
        plt.close("all")


if __name__ == "__main__":
    main()
