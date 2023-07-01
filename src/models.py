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
        arch: str = "Unet",
    ) -> None:
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=3,
            classes=1,
            activation=None,
            aux_params=aux_params,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            logits, cls_logits = outputs
            cls_logits = cls_logits.reshape(-1)
        else:
            logits = outputs
            cls_logits = None

        if logits.shape[1] != 256:
            logits = nn.functional.interpolate(
                logits, size=(256, 256), mode="bilinear", align_corners=False
            )
        logits = logits.squeeze(1)

        # logist: (batch_size, height, width)
        outputs = {
            "logits": logits,
            "cls_logits": cls_logits,
        }
        return outputs
