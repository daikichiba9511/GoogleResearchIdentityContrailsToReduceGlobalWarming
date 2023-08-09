from datetime import datetime, timedelta
from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Any, Final, Sequence, TypeVar

import cv2
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["get_stream_logger", "add_file_handler"]

logger = getLogger(__name__)

DEFAULT_FORMAT: Final[
    str
] = "[%(levelname)s] %(asctime)s - %(pathname)s : %(lineno)d : %(funcName)s : %(message)s"


def get_stream_logger(level: int = INFO, format: str = DEFAULT_FORMAT) -> Logger:
    logger = getLogger()
    logger.setLevel(level)

    handler = StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)
    return logger


def add_file_handler(
    logger: Logger, filename: str, level: int = INFO, format: str = DEFAULT_FORMAT
) -> None:
    handler = FileHandler(filename=filename)
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)


def is_nan(x: np.ndarray | float) -> bool:
    if isinstance(x, np.ndarray):
        return bool(np.isnan(x).any())

    return np.isnan(x)


def get_called_time() -> str:
    """Get current time in JST (Japan Standard Time = UTC+9)"""
    now = datetime.utcnow() + timedelta(hours=9)
    return now.strftime("%Y%m%d%H%M%S")


_T = TypeVar("_T")


def flatten_dict(values: Sequence[dict[str, _T]]) -> dict[str, _T]:
    flattend = {}
    for value in values:
        flattend.update(value)
    return flattend


def list_to_string(x: list[int]) -> str:
    if x:
        return str(x).replace("[", "").replace("]", "").replace(",", "")
    return "-"


def rle_encode(preds: np.ndarray, fg_val: int = 1) -> str:
    """
    Args:
        preds (np.ndarray): Predictions of shape (H, W), 1 - mask, 0 - background
    """
    dots = np.where(preds.T.flatten() == fg_val)[0]
    run_lengths: list[int] = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return list_to_string(run_lengths)


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Args:
        mask_rle (str): Run-length as string formated (start length)
        shape (tuple[int, int]): (height, width) of array to return
    Returns:
        np.ndarray: 1 - mask, 0 - background
    """
    image = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    if mask_rle != "-":
        s = mask_rle.split()
        starts, length = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + length
        for lo, hi in zip(starts, ends):
            image[lo:hi] = 1
    # Fortran like index ordering
    return image.reshape(shape, order="F")


def plot_preds(
    pred: np.ndarray, image: np.ndarray, label: np.ndarray, threshold: float = 0.5
) -> tuple:
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(pred)
    ax[0].set_title("Pred")
    ax[1].imshow(pred > threshold)
    ax[1].set_title(f"Contrails {threshold}")
    ax[2].imshow(image)
    ax[2].set_title("Image")
    ax[3].imshow(label)
    ax[3].set_title("Label")
    return fig, ax


def plot_preds_with_label_on_image(
    pred: np.ndarray,
    image: np.ndarray,
    label: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
) -> tuple:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if not isinstance(ax, plt.Axes):
        raise ValueError("image must be plt.Axes")

    color_label = np.zeros((*label.shape, 3))
    color_label[label == 1] = (0, 1, 0)

    color_pred = np.zeros((*pred.shape, 3))
    color_pred[pred > 0.5] = (1, 0, 0)

    ax.imshow(image)
    ax.imshow(color_pred, alpha=0.5, label="pred")
    ax.imshow(color_label, alpha=0.5, label="label")

    return fig, ax


def filter_tiny_objects(image: np.ndarray, thr: int) -> np.ndarray:
    _image = image.copy()
    # _image = image

    num_labels, labels = cv2.connectedComponents(_image.astype(np.uint8))
    for label in range(1, num_labels):
        if np.sum(labels == label) < thr:
            _image[labels == label] = 0
    return _image
