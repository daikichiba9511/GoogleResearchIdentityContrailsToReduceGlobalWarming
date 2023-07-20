from enum import Enum
from logging import getLogger
from typing import Any, Literal, TypeAlias

import torch
import torch.nn as nn
import torch.optim as optim
from typing_extensions import assert_never

logger = getLogger(__name__)

__all__ = [
    "OptimizerTypeStr",
    "OptimizerType",
    "get_optimizer",
]

OptimizerTypeStr: TypeAlias = Literal["sgd", "adam", "adamw"]


class OptimizerType(str, Enum):
    SGD = "sgd"
    Adam = "adam"
    AdamW = "adamw"


def get_optimizer_params(
    model: nn.Module, encoder_lr: float, decorder_lr: float
) -> list[dict[str, object]]:
    params = model.named_parameters()
    encoder_params = list(
        map(lambda x: x[1], filter(lambda x: "encoder" in x[0], params))
    )
    decorder_params = list(
        map(lambda x: x[1], filter(lambda x: "encoder" not in x[0], params))
    )
    optimizer_params = [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": decorder_params, "lr": decorder_lr},
    ]
    return optimizer_params


def get_optimizer(
    optimizer_type: OptimizerTypeStr | OptimizerType,
    optimizer_params: dict[str, Any],
    model: nn.Module,
) -> optim.Optimizer:
    """Get optimizer

    Args:
        optimizer_type (str | OptimizerType): Optimizer type
        optimizer_params (dict[str, object]): Optimizer parameters
        model (nn.Module): Model

    Raises:
        ValueError: If optimizer type is not supported

    Returns:
        optim.Optimizer: Optimizer
    """
    optimizer_type = OptimizerType(optimizer_type)
    if optimizer_params.get("encoder_lr") and optimizer_params.get("decoder_lr"):
        logger.info("Use different learning rates for encoder and decorder")
        model_parameters = get_optimizer_params(
            model=model,
            encoder_lr=optimizer_params.pop("encoder_lr"),
            decorder_lr=optimizer_params.pop("decoder_lr"),
        )
    else:
        model_parameters = model.parameters()

    match optimizer_type:
        case OptimizerType.SGD:
            optimizer = optim.SGD(model_parameters, **optimizer_params)
            return optimizer
        case OptimizerType.Adam:
            optimizer = optim.Adam(model_parameters, **optimizer_params)
            return optimizer
        case OptimizerType.AdamW:
            optimizer = optim.AdamW(model_parameters, **optimizer_params)
            return optimizer
        case _:
            assert_never(optimizer_type)


def _test_get_optimizer() -> None:
    optimizer = get_optimizer(
        optimizer_type="adam", optimizer_params={"lr": 0.01}, model=nn.Linear(10, 10)
    )
    print(optimizer)

    optimizer = get_optimizer(
        optimizer_type=OptimizerType.AdamW,
        optimizer_params={"lr": 0.01},
        model=nn.Linear(10, 10),
    )
    print(optimizer)

    try:
        _ = get_optimizer(
            optimizer_type="error",  # type: ignore
            optimizer_params={"lr": 0.01},
            model=nn.Linear(10, 10),
        )
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    _test_get_optimizer()
