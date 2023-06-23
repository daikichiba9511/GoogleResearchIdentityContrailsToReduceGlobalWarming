from logging import INFO, FileHandler, Formatter, Logger, StreamHandler, getLogger
from typing import Final

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
) -> Logger:
    handler = FileHandler(filename=filename)
    handler.setLevel(level)
    handler.setFormatter(Formatter(format))
    logger.addHandler(handler)


def list_to_string(x: list[object]) -> str:
    if x:
        return str(x).replace("[", "").replace("]", "").replace(",", "")
    return "-"


def rle_encode(preds, fg_val: int = 1) -> str:
    """
    Args:
        preds (np.ndarray): Predictions of shape (H, W), 1 - mask, 0 - background
    """
    dots = np.where(preds.T.flatten() == fg_val)[0]
    run_lengths = []
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
