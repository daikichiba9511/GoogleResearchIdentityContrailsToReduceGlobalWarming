import importlib
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from pprint import pformat
from typing import TypeVar

from src.optimizer import OptimizerType, OptimizerTypeStr
from src.scheduler import SchedulerType, SchedulerTypeStr

logger = getLogger(__name__)

T = TypeVar("T")


@dataclass
class Config:
    expname: str
    description: str
    seed: int

    # -- Model
    arch: str
    encoder_name: str
    encoder_weight: str | None
    checkpoints: list[str]

    # -- Data
    data_root_path: Path
    train_csv_path: Path
    valid_csv_path: Path

    image_size: int
    n_splits: int

    # -- Training
    train_batch_size: int
    valid_batch_size: int
    output_dir: Path
    train_params: dict[str, int | float]

    epochs: int
    patience: int

    loss_type: str
    loss_params: dict[str, int | float]

    optimizer_type: OptimizerTypeStr | OptimizerType
    optimizer_params: dict[str, int | float]

    scheduler_type: SchedulerTypeStr | SchedulerType
    scheduler_params: dict[str, int | float]

    # -- Inference
    test_batch_size: int
    threshold: float


def init_config(config_cls: T, config_path: str) -> T:
    """Load config from config_path and initialize config_cls

    Args:
        config_cls (T): Config class for validation
        config_path (str): Config path
    """
    _config = importlib.import_module(config_path).config
    logger.info(f"Loaded config from {config_path}")
    # logger.info(f"\n\tConfig: \n{pformat(_config)}")
    return config_cls(**_config)
