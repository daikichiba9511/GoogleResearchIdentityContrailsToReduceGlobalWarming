from enum import Enum
from typing import Literal, TypeAlias

import torch.nn as nn
import torch.optim as optim
from typing_extensions import assert_never

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


def get_optimizer(
    optimizer_type: OptimizerTypeStr | OptimizerType,
    optimizer_params: dict[str, object],
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
            optimizer_type="error",
            optimizer_params={"lr": 0.01},
            model=nn.Linear(10, 10),
        )
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    _test_get_optimizer()
