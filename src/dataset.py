from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def read_record(
    dir: Path, load_band_files: list[str] = ["band_11", "band_14", "band_15"]
) -> dict[str, np.ndarray]:
    data = {}
    for band in load_band_files:
        data[band] = np.load(dir / f"{band}.npy")
        # data[band] = self.load_img(str(dir / f"{band}.npy"))
    return data


def normalize_range(
    data: np.ndarray, bounds: tuple[float | int, float | int]
) -> np.ndarray:
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def get_false_color(record_data: dict[str, np.ndarray]) -> np.ndarray:
    _t11_bounds = (243, 303)
    _cloud_tp_tdiff_bounds = (-4, 5)
    _tdiff_bounds = (-4, 2)

    r = normalize_range(record_data["band_15"] - record_data["band_14"], _tdiff_bounds)
    g = normalize_range(
        record_data["band_14"] - record_data["band_11"], _cloud_tp_tdiff_bounds
    )
    b = normalize_range(record_data["band_14"], _t11_bounds)

    false_color: np.ndarray = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    # shape: (256, 256, T), T = n_times_before + n_times_after + 1 = 8
    return false_color


class ContrailsDataset(Dataset):
    """ContrailsDataset

    Args:
        df (pd.DataFrame): dataframe of contrails dataset containing path to image files.
        image_size (int, optional): image size. Defaults to 224.
        train (bool, optional): whether to use train dataset. Defaults to True.
        normalize_fn (Callable, optional): normalization function. Defaults to None.

    NOTE:
        trainのときは前処理でRGB画像にしてる公開データセットを使う
        ラベルがついてるtimestampの画像のみつかってることに注意する

    References:
    [1] https://www.kaggle.com/code/egortrushin/gr-icrgw-training-with-4-folds
    [2] https://www.kaggle.com/datasets/shashwatraman/contrails-images-ash-color
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = True,
        normalize_fn: Callable | None = None,
        transform_fn: Callable | None = None,
    ) -> None:
        self.df = df
        self.image_size = image_size
        self.is_train = train

        if normalize_fn is None:
            # self.normalize_image = T.Normalize(
            #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            # )
            self.normalize_image = A.Normalize()
        else:
            self.normalize_image = normalize_fn

        if image_size != 256:
            self.resize_image = T.transforms.Resize(256, antialias=True)
        else:
            self.resize_image = None

        self.transform_fn = transform_fn

    def load_img(self, path: str) -> np.ndarray:
        file = open(path, "rb")
        header = file.read(128)
        desc = str(header[15:25], "utf-8").replace("'", "").replace(" ", "")
        shape = tuple(
            int(num)
            for num in str(header[60:120], "utf-8")
            .replace(", }", "")
            .replace("(", "")
            .replace(")", "")
            .split(",")
        )
        datasize = np.lib.format.descr_to_dtype(desc).itemsize
        for dimension in shape:
            datasize *= dimension
        return np.ndarray(shape=shape, dtype=desc, buffer=file.read(datasize))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        if self.is_train:
            contrails_image_path = row["path"]
            # shape: (256, 256, T), T = n_times_before + n_times_after + 1 = 8
            # n_times_before = 4, n_times_after = 3
            contrails_image = np.load(str(contrails_image_path))
            # contrails_image = self.load_img(str(contrails_image_path))
            raw_image = contrails_image[..., :-1]
            raw_label = contrails_image[..., -1]

            raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)
            raw_label = np.reshape(raw_label, (256, 256)).astype(np.float32)

            if self.transform_fn is not None:
                augmented = self.transform_fn(image=raw_image, mask=raw_label)
                image = augmented["image"]
                label = augmented["mask"]
            else:
                image = torch.tensor(raw_image).float().permute(2, 0, 1)
                label = torch.tensor(raw_label).float()

            if self.image_size != 256 and self.resize_image is not None:
                label = TF.resize(
                    label.unsqueeze(0), size=[256, 256], antialias=True
                ).squeeze(0)

            return image, label

        else:
            contrails_image_path = row["path"]
            record_data = read_record(contrails_image_path)
            n_times_before = 4
            false_color_img = get_false_color(record_data)
            raw_image = false_color_img[..., n_times_before]
            raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)

            if self.transform_fn is not None:
                augmented = self.transform_fn(image=raw_image)
                image = augmented["image"]
            else:
                image = torch.tensor(raw_image).float().permute(2, 0, 1)

            image_id = self.df.iloc[index]["record_id"]
            image_id = torch.tensor(int(image_id))

            return image, image_id

    def __len__(self) -> int:
        return len(self.df)


class ClsDataset(Dataset):
    def __init__(
        self, img_dirs: Sequence[Path], transform_fn: Callable | None = None
    ) -> None:
        self.img_dirs = img_dirs
        self.transform_fn = transform_fn

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        img_dir = self.img_dirs[index]
        record_data = read_record(img_dir)
        # shape: (256, 256, 3, T), T = n_times_before + n_times_after + 1 = 8
        false_color = get_false_color(record_data)
        n_times_before = 4

        imgs, image_ids = [], []
        for i in range(8):
            if i == n_times_before:
                continue
            raw_image = false_color[..., i]
            raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)

            if self.transform_fn is not None:
                augmented = self.transform_fn(image=raw_image)
                image = augmented["image"]
            else:
                image = torch.tensor(raw_image).float().permute(2, 0, 1)

            image_id = str(img_dir.stem) + "_" + str(i)

            image_ids.append(image_id)
            imgs.append(image)

        return torch.stack(imgs), image_ids

    def __len__(self) -> int:
        return len(self.img_dirs)


if __name__ == "__main__":
    root = Path("./input/google-research-identify-contrails-reduce-global-warming")
    img_dirs = list((root / "train").glob("*"))[:100]
    print(len(img_dirs))
    cls_dataset = ClsDataset(img_dirs=img_dirs)
    print(len(cls_dataset))
    imgs, image_ids = cls_dataset[0]
    print(imgs.shape)
    print(image_ids)
