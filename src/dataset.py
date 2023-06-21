from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class ContrailsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        train: bool = True,
        normalize_fn: Callable | None = None,
    ) -> None:
        self.df = df
        self.image_size = image_size
        self.is_train = train

        if normalize_fn is None:
            self.normalize_image = T.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            )
        else:
            self.normalize_image = normalize_fn

        if image_size != 256:
            self.resize_image = T.transforms.Resize((image_size, image_size))
        else:
            self.resize_image = None

    def read_record(
        self, dir: Path, load_band_files: list[str] = ["band_11", "band_14", "band_15"]
    ) -> dict[str, np.ndarray]:
        data = {}
        for band in load_band_files:
            data[band] = np.load(dir / f"{band}.npy")
        return data

    def normalize_range(
        self, data: np.ndarray, bounds: tuple[float | int, float | int]
    ) -> dict[str, np.ndarray]:
        return (data - bounds[0]) / (bounds[1] - bounds[0])

    def get_false_color(self, record_data: dict[str, np.ndarray]) -> np.ndarray:
        _t11_bounds = (243, 303)
        _cloud_tp_tdiff_bounds = (-4, 5)
        _tdiff_bounds = (-4, 2)

        n_times_before = 4

        r = self.normalize_range(
            record_data["band_15"] - record_data["band_14"], _tdiff_bounds
        )
        g = self.normalize_range(
            record_data["band_14"] - record_data["band_11"], _cloud_tp_tdiff_bounds
        )
        b = self.normalize_range(record_data["band_14"], _t11_bounds)

        face_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
        image = face_color[..., n_times_before]
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        contrails_image_path = row["path"]
        if self.is_train:
            contrails_image = np.load(str(contrails_image_path))
            # TODO: なんで-1をつかってるか調べる
            image = contrails_image[..., :-1]
            label = contrails_image[..., -1]
            label = torch.tensor(label).float()
            image = (
                torch.tensor(np.reshape(image, (256, 256, 3))).float().permute(2, 0, 1)
            )
            if self.image_size != 256 and self.resize_image_fn is not None:
                image = self.resize_image(image)

            image = self.normalize_image(image)
            return image, label
        else:
            record_data = self.read_record(contrails_image_path)
            image = self.get_false_color(record_data)
            image = (
                torch.tensor(np.reshape(image, (256, 256, 3))).float().permute(2, 0, 1)
            )
            image_id = self.df.iloc[index]["record_id"]
            image_id = torch.tensor(int(image_id))
            if self.image_size != 256 and self.resize_image_fn is not None:
                image = self.resize_image(image)

            image = self.normalize_image(image)
            return image, image_id

    def __len__(self) -> int:
        return len(self.df)
