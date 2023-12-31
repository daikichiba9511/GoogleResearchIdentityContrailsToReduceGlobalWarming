from enum import Enum
from typing import Any, Literal, TypeAlias

import torch.nn as nn
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from typing_extensions import assert_never

__all__ = [
    "SchedulerTypeStr",
    "SchedulerType",
    "get_scheduler",
]

SchedulerTypeStr: TypeAlias = Literal[
    "cosine_with_warmup", "reduce_on_plateau", "cosingannealinglr"
]


class SchedulerType(str, Enum):
    CosineWithWarmup = "cosine_with_warmup"
    ReduceOnPlateau = "reduce_on_plateau"
    CosineAnnealingLR = "cosineannealinglr"


def get_scheduler(
    scheduler_type: SchedulerTypeStr | SchedulerType,
    scheduler_params: dict[str, Any],
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.LRScheduler:
    scheduler_type = SchedulerType(scheduler_type)
    match scheduler_type:
        case SchedulerType.CosineWithWarmup:
            scheduler = CosineLRScheduler(optimizer=optimizer, **scheduler_params)
            return scheduler  # type: ignore
        case SchedulerType.ReduceOnPlateau:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, **scheduler_params
            )

            return scheduler  # type: ignore
        case SchedulerType.CosineAnnealingLR:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, **scheduler_params
            )
            return scheduler
        case _:
            assert_never(scheduler_type)


def _test_get_scheduler() -> None:
    scheduler = get_scheduler(
        "cosine_with_warmup",
        {"num_warmup_steps": 100, "num_training_steps": 1000},
        optim.AdamW(nn.Linear(10, 10).parameters()),
    )
    print(scheduler)
    print(scheduler.get_lr())

    scheduler = get_scheduler(
        SchedulerType.CosineWithWarmup,
        {"num_warmup_steps": 100, "num_training_steps": 1000},
        optim.AdamW(nn.Linear(10, 10).parameters()),
    )
    print(scheduler)
    print(scheduler.get_lr())

    try:
        _ = get_scheduler(
            "### error!!!! ###",  # type: ignore
            {"num_warmup_steps": 100, "num_training_steps": 1000},
            optim.AdamW(nn.Linear(10, 10).parameters()),
        )
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    _test_get_scheduler()
