import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class ContrailsModel(nn.Module):
    def __init__(self, encoder_name: str, encoder_weight: str | None = None) -> None:
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weight,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.model(images)

        outputs = {
            "logits": logits,
        }
        return outputs