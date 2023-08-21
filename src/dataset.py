from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Sequence, TypeVar

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from tqdm.auto import tqdm


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


_T = TypeVar("_T", np.ndarray, torch.Tensor)


def rescale_range(img: _T, min_value: float, max_value: float) -> _T:
    return img - min_value / (max_value - min_value)


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


@lru_cache(maxsize=128)
def _load_image(path: str) -> tuple[np.ndarray, np.ndarray]:
    contrails_image = np.load(str(path))
    raw_image = contrails_image[..., :-1]
    raw_label = contrails_image[..., -1]
    raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)
    raw_label = np.reshape(raw_label, (256, 256)).astype(np.float32)
    return raw_image, raw_label


def grid_img(pts_per_grid: int, offset: float = 0.5) -> torch.Tensor:
    """create xy values of grid_num x grid_num grid

    Args:
        pts_per_grid (int): number of grid points per dimension (e.g 512)
        offset (float, optional): offset of grid. Defaults to 0.5.

    Returns:
        torch.Tensor: shape = (pts_per_grid, pts_per_grid, 2)
    """
    grid = np.zeros((pts_per_grid, pts_per_grid, 2), dtype=np.float32)
    for ix in range(pts_per_grid):
        for iy in range(pts_per_grid):
            grid[ix, iy, 1] = -1 + 2 * (ix + 0.5) / pts_per_grid + offset / 128
            grid[ix, iy, 0] = -1 + 2 * (iy + 0.5) / pts_per_grid + offset / 128
    grid = torch.from_numpy(grid)
    return grid


def soft_label(labels: np.ndarray, weight: int = 1) -> np.ndarray:
    """make_soft_label

    Args:
        labels (np.ndarray): shape = (H, W, 1, R)

    Returns:
        np.ndarray: shape = (H, W)

    References:
    https://github.com/tattaka/google-research-identify-contrails-reduce-global-warming/blob/main/src/exp055/train_stage1_seg.py#L112
    """
    h, w, _, r = labels.shape
    soft_labels = np.clip((weight * labels).sum(axis=-1) / r, 0, 1)
    return soft_labels.reshape(h, w)


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
        image_paths: Sequence[Path],
        image_size: int = 224,
        train: bool = True,
        normalize_fn: Callable | None = None,
        transform_fn: Callable | None = None,
        image_ids: Sequence[str] | None = None,
        use_soft_label: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.image_ids = image_ids
        self.image_size = image_size
        self.is_train = train
        self.use_soft_label = use_soft_label

        if normalize_fn is None:
            # self.normalize_image = T.Normalize(
            #     (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            # )
            self.normalize_image = A.Normalize()
        else:
            self.normalize_image = normalize_fn

        if image_size != 256:
            self.resize_image = T.transforms.Resize(256, antialias=True)  # type: ignore
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
        if self.is_train:
            contrails_image_path = self.image_paths[index]
            # shape: (256, 256, T), T = n_times_before + n_times_after + 1 = 8
            # n_times_before = 4, n_times_after = 3
            # contrails_image = np.load(str(contrails_image_path))
            # contrails_image = self.load_img(str(contrails_image_path))
            # raw_image = contrails_image[..., :-1]
            # raw_label = contrails_image[..., -1]

            # raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)
            # raw_label = np.reshape(raw_label, (256, 256)).astype(np.float32)
            raw_image, raw_label = _load_image(contrails_image_path)

            if self.transform_fn is not None:
                augmented = self.transform_fn(image=raw_image, mask=raw_label)
                image = augmented["image"]
                label = augmented["mask"]
                # label = torch.tensor(raw_label).float()
            else:
                image = torch.tensor(raw_image).float().permute(2, 0, 1)
                label = torch.tensor(raw_label).float()

            # if self.image_size != 256:
            #     label = TF.resize(
            #         label.unsqueeze(0), size=[256, 256], antialias=True
            #     ).squeeze(0)

            return image, label

        else:
            contrails_image_path = self.image_paths[index]
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

            if self.image_ids is None:
                raise ValueError("image_ids must be set when is_train=False")

            image_id = self.image_ids[index]
            image_id = torch.tensor(int(image_id))
            return image, image_id

    def __len__(self) -> int:
        return len(self.image_paths)


def fixed_offset_img(
    img: np.ndarray, img_size: tuple[int, int] = (512, 512)
) -> np.ndarray:
    """fix offset of an image which is caused by polygon2mask

    NOTE:
        - Affine transformation

        [x, y, 1] = [[a, b, t_x], [c, d, t_y], [0, 0, 0]] @ [x', y', 1]

        where.

        (tx, ty) is for translation.

        (a, b, c, d) is for rotation and scaling.

        If you want to know about affine transformation, please refer [1] and [2].
        (specifically, [2] is very easy to understand the theory of affine transformation)

    References:
    [1] https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430479
    [2] https://note.nkmk.me/python-opencv-warp-affine-perspective/
    [3] https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/discussion/430749
    """
    # If you want to upsample 2x in x direction, 1.5x in y direction, you should use M = [[2.0, 0.0, 0.0], [0.0, 1.5, 0.0]]
    # If you want to move 25px in x direction, 50px in y direction, you should use M = [[1.0, 0.0, 25], [0.0, 1.0, 50]]
    # When image size is 512x512, you should translate 1.5px in both x and y directions.
    # So, you should use M = [[2.0, 0.0, 1.5], [0.0, 2.0, 1.5]]
    #
    # Make the image size (256, 256), 0.5px offset in both x and y directions occurs because of the translation of the polygon2mask using opencv(probably?)
    # So, there are two possibilities. -0.5px offset or 0.5px offset. Since there is the offset, left-top or right-bottom was not included.
    # If you want to know the offset, please refer [1] and [3]

    img_affine_matrix = np.array(
        [
            [2.0, 0.0, 1.5],
            [0.0, 2.0, 1.5],
        ],
        dtype=np.float64,
    )
    fixed_img = cv2.warpAffine(
        img,
        img_affine_matrix,
        img_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return fixed_img


class ContrailsDatasetV2(Dataset):
    def __init__(
        self,
        img_paths: Sequence[Path],
        transform_fn: Callable | None = None,
        phase: str = "train",
        use_soft_label: bool = True,
        mask_paths: Sequence[Path] | None = None,
        avg_mask_paths: Sequence[Path] | None = None,
    ) -> None:
        if phase not in ["train", "val", "test"]:
            raise ValueError(f"phase must be one of train, val, test, but got {phase}")

        self.img_paths = img_paths
        self.transform_fn = transform_fn
        self.phase = phase
        self.use_soft_label = use_soft_label
        self.mask_paths = mask_paths
        self.avg_mask_paths = avg_mask_paths
        self.grid = grid_img(512, offset=0.5)

    def __len__(self) -> int:
        return len(self.img_paths)

    def _transform(
        self, image: np.ndarray, mask: np.ndarray | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if image.ndim != 3:
            raise ValueError(f"image must be 3 dimension, but got {image.ndim}")
        if mask is not None and mask.ndim != 2:
            raise ValueError(f"mask must be 2 dimension, but got {mask.ndim}")

        if self.transform_fn is not None and mask is not None:
            augmented = self.transform_fn(image=image, mask=mask)
            return augmented["image"], augmented["mask"]
            # return augmented["image"], torch.from_numpy(mask).float()

        elif self.transform_fn is None and mask is not None:
            _image = torch.tensor(image).float().permute(2, 0, 1)
            _target = torch.tensor(mask).float()
            return _image, _target
        else:
            _image = torch.tensor(image).float().permute(2, 0, 1)
            return _image, None

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_path = self.img_paths[index]

        match self.phase:
            case "train":
                if self.mask_paths is None:
                    raise ValueError("mask_paths must be set when phase=train")
                if self.avg_mask_paths is None:
                    raise ValueError("avg_mask_paths must be set when phase=train")

                mask_path = self.mask_paths[index]
                avg_mask_path = self.avg_mask_paths[index]

                raw_image = np.load(img_path).astype(np.float32)
                # raw_image = fixed_offset_img(raw_image)
                pixel_mask = np.load(mask_path).astype(np.float32)
                avg_mask = np.load(avg_mask_path).astype(np.float32)

                mask = avg_mask if self.use_soft_label else pixel_mask
                if raw_image.shape != (256, 256, 3):
                    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_LINEAR)

                image, target = self._transform(raw_image, mask)
                if target is None:
                    raise ValueError("target must not be None")

                return {
                    "image": image,
                    "pixel_mask": torch.tensor(pixel_mask),
                    "target": target,
                }

            case "val":
                if self.mask_paths is None:
                    raise ValueError("mask_paths must be set when phase=train")

                mask_path = self.mask_paths[index]
                raw_image = np.load(img_path).astype(np.float32)
                # raw_image = fixed_offset_img(raw_image)
                pixel_mask = np.load(mask_path).astype(np.float32)

                if raw_image.shape != (256, 256, 3):
                    pixel_mask = cv2.resize(
                        pixel_mask, (512, 512), interpolation=cv2.INTER_LINEAR
                    )

                image, target = self._transform(raw_image, pixel_mask)
                if target is None:
                    raise ValueError("target must not be None")
                return {"image": image, "target": target}

            case "test":
                record_data = read_record(img_path)
                n_times_before = 4
                raw_image = (
                    get_false_color(record_data)[..., n_times_before]
                    .reshape(256, 256, 3)
                    .astype(np.float32)
                )
                # raw_image = fixed_offset_img(raw_image)

                pixel_mask = (
                    np.load(img_path / "human_pixel_masks.npy")
                    .reshape(256, 256)
                    .astype(np.float32)
                )
                image, _ = self._transform(raw_image)
                return {"image": image}

            case _:
                raise ValueError(
                    f"phase must be one of train, val, test, but got {self.phase}"
                )


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

        imgs, img_ids = [], []
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

            img_ids.append(image_id)
            imgs.append(image)

        return torch.stack(imgs), img_ids

    def __len__(self) -> int:
        return len(self.img_dirs)


class SegDataset(Dataset):
    def __init__(
        self,
        img_dirs: Sequence[Path],
        transform_fn: Callable | None = None,
        skip_image_ids: Sequence[str] | None = None,
    ) -> None:
        self.img_dirs = img_dirs
        self.transform_fn = transform_fn

        n_times_before = 4

        imgs, img_ids = [], []
        for img_dir in tqdm(
            self.img_dirs,
            desc="Loading data...",
            total=len(self.img_dirs),
            dynamic_ncols=True,
        ):
            record_data = read_record(img_dir)
            # shape: (256, 256, 3, T), T = n_times_before + n_times_after + 1 = 8
            false_color = get_false_color(record_data)
            for i in range(8):
                if i == n_times_before:
                    continue
                raw_image = false_color[..., i]
                raw_image = np.reshape(raw_image, (256, 256, 3)).astype(np.float32)

                image_id = str(img_dir.stem) + "_" + str(i)
                if skip_image_ids is not None and image_id in skip_image_ids:
                    continue

                img_ids.append(image_id)
                imgs.append(raw_image)
        self.imgs = imgs
        self.img_ids = img_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        img = self.imgs[index]
        img_id = self.img_ids[index]

        if self.transform_fn is not None:
            augmented: dict[str, torch.Tensor] = self.transform_fn(image=img)
            image: torch.Tensor = augmented["image"]
        else:
            image = torch.tensor(img).float().permute(2, 0, 1)
        return image, img_id

    def __len__(self) -> int:
        return len(self.img_dirs)


if __name__ == "__main__":
    root = Path("./input/google-research-identify-contrails-reduce-global-warming")
    img_dirs = list((root / "train").glob("*"))[:100]
    print(len(img_dirs))
    cls_dataset = ClsDataset(img_dirs=img_dirs)
    print(len(cls_dataset))
    batch = cls_dataset[0]
    # print(batch)

    import matplotlib.pyplot as plt
    from albumentations.pytorch import ToTensorV2

    img = np.load(Path("./input/prepared_np_imgs_weight1/195731008142151/image.npy"))
    print(img.shape)
    fixed_img = fixed_offset_img(img)
    print(fixed_img.shape)

    img = (
        T.Resize(512, antialias=True)(ToTensorV2()(image=img)["image"])  # type: ignore
        .permute(1, 2, 0)
        .numpy()
    )
    fig, (axes_0, axes_1) = plt.subplots(1, 2, figsize=(10, 5))
    # assert isinstance(axes, plt.Axes)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes_0, plt.Axes)
    assert isinstance(axes_1, plt.Axes)

    axes_0.imshow(img, alpha=0.5)
    axes_0.set_xticks(np.arange(0, 512, 64))  # type: ignore
    axes_0.set_yticks(np.arange(0, 512, 64))  # type: ignore
    axes_0.grid(which="both", color="white", linewidth=2)

    axes_1.imshow(fixed_img, alpha=0.5)
    axes_1.set_yticks(np.arange(0, 512, 64))  # type: ignore
    axes_1.set_xticks(np.arange(0, 512, 64))  # type: ignore
    axes_1.grid(which="both", color="white", linewidth=2)

    fig.savefig("./output/eda/fixed_offset.png")
