from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ContrailsModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weight: str | None = None,
        aux_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=3,
            classes=1,
            activation=None,
            aux_params=aux_params,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits, cls_logits = self.model(images)

        # logist: (batch_size, height, width)
        outputs = {
            "logits": logits.squeeze(1),
            "cls_logits": cls_logits.reshape(-1),
        }
        return outputs
