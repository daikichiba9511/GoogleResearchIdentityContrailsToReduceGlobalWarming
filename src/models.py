from logging import getLogger
from typing import Any

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unetr import UNETR
from transformers import SegformerForSemanticSegmentation

logger = getLogger(__name__)


class ContrailsModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weight: str | None = None,
        aux_params: dict[str, Any] | None = None,
        arch: str = "Unet",
    ) -> None:
        super().__init__()
        encoder_depth = 5 if not encoder_name.startswith("tu-convnext") else 4
        decoder_channels = (
            [256, 128, 64, 32, 16] if encoder_depth == 5 else [256, 128, 64, 32]
        )

        if arch == "UNet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weight,
                encoder_depth=encoder_depth,
                decoder_channels=decoder_channels,
                in_channels=3,
                classes=1,
                activation=None,
                aux_params=aux_params,
            )
        if arch == "SwinUNETR":
            self.model = SwinUNETR(
                img_size=(512, 512),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,
                use_v2=False,
                use_checkpoint=True,
            )

        else:
            self.model = smp.create_model(
                arch=arch,
                encoder_name=encoder_name,
                encoder_weights=encoder_weight,
                encoder_depth=encoder_depth,
                decoder_channels=decoder_channels,
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

        preds = nn.functional.interpolate(
            logits, size=(256, 256), mode="bilinear", align_corners=False
        )
        # preds = preds.squeeze(1)
        # logits = logits.squeeze(1)

        # logist: (batch_size, height, width)
        outputs = {
            "logits": logits,
            "cls_logits": cls_logits,
            "preds": preds,
        }
        return outputs


class Residual3DBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm3d(num_features=512),
        )

    def forward(self, x):
        shortcut = x
        h = self.block(x)
        h = self.block2(h)
        out = F.relu(h + shortcut)
        return out


class GeM(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(
        self, x: torch.Tensor, p: int | nn.Parameter = 3, eps: float = 1e-6
    ) -> torch.Tensor:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ContrailsModelV2(nn.Module):
    """
    Reference:
    [1] https://www.kaggle.com/competitions/nfl-player-contact-detection/discussion/392402#2170010
    """

    def __init__(self, backbone: str = "tf_efficientnet_b0.ns_jft_in1k") -> None:
        super().__init__()
        # 2dcnn
        self.backbone = timm.create_model(
            backbone, pretrained=True, num_classes=1, in_chans=3
        )
        self.mlp = nn.Sequential(
            nn.Linear(68, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        # pointwise conv2d
        n_hidden = 1024
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=512, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
        )

        self.neck = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
        )

        # 3dcnn
        self.triple_layer = nn.Sequential(
            Residual3DBlock(),
        )

        self.pool = GeM()

        # self.fc = nn.Linear(256 + 1024, 1)
        self.fc = nn.Linear(1024, 1)

    def forward(
        self,
        images: torch.Tensor,
        feature: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, h, w = images.shape
        # 後でin_chans=3にfeature_forwardに通すため
        # batch側に寄せてencodeする
        images = images.view(b * t // 3, 3, h, w)
        # encoded_feature: (80, 1280, 16, 16)
        encoded_feature = self.backbone.forward_features(images)
        feature_maps = self.conv_proj(encoded_feature)
        _, c, h, w = feature_maps.shape
        feature_maps = feature_maps.contiguous().view(b * 2, c, t // 2 // 3, h, w)
        feature_maps = self.triple_layer(feature_maps)
        # middle_maps: (16, 512, 5, 16, 16)
        # 真ん中を取り出すのは対象のフレーム<=>着目してるフレーム
        middle_maps = feature_maps[:, :, 2, :, :]

        # 抽出した着目してるフレームの特徴量をpoolingすることでフレーム内のコンテキストの情報を集約する
        # pooled_maps: (16, 512, 1, 1)
        pooled_maps = self.pool(middle_maps)
        # reshpaed_pooled_maps: (8, 512*2)
        nn_feature = self.neck(pooled_maps.reshape(b, -1))

        # 単に特徴から学習につかう特徴を抽出する
        if feature is not None:
            mlp_feature = self.mlp(feature)
            cat_feature = torch.cat([nn_feature, mlp_feature], dim=1)
        else:
            cat_feature = nn_feature
        y = self.fc(cat_feature)
        return y


class UNETR_Segformer(nn.Module):
    def __init__(self, img_size: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        # img_size: (depth, height, width)
        self.encoder = UNETR(
            in_channels=3,
            out_channels=32,
            # img_size=(16, img_size[0], img_size[1]),
            img_size=img_size,
            spatial_dims=2,  # H, W
            conv_block=True,
        )
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=1,
            ignore_mismatched_sizes=True,
            num_channels=32,
        )
        self.upscaler1 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1
        )
        self.upscaler2 = nn.ConvTranspose2d(
            1, 1, kernel_size=(4, 4), stride=2, padding=1
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            image: (batch, c, depth, h, w)
        """
        output = self.encoder(image)
        output = self.dropout(output)  # (b, num_channels, img_size, img_size)
        output = self.encoder_2d(output).logits  # type: ignore
        output = self.upscaler1(output)
        output = self.upscaler2(output)

        preds = F.interpolate(
            output, size=(256, 256), mode="bilinear", align_corners=False
        ).squeeze(1)
        output = output.squeeze(1)

        outputs = {
            "logits": output,
            "preds": preds,
        }
        return outputs


if __name__ == "__main__":
    # model = ContrailsModelV2()
    # im = torch.randn(2, 3 * 8, 256, 256)
    # out = model(im)
    # print(out.shape)
    #
    # model = UNETR_Segformer(img_size=256)
    # im = torch.randn(2, 3, 256, 256)
    # out = model(im)
    # print(out["logits"].shape)
    # print(out["preds"].shape)
    #
    # model = UNETR_Segformer(img_size=512).cuda()
    # im = torch.randn(4, 3, 512, 512).cuda()
    # out = model(im)
    # print(out["logits"].shape)
    # print(out["preds"].shape)

    model = ContrailsModel(encoder_name="swinv2", arch="SwinUNETR")
    im = torch.randn(8, 3, 512, 512)
    out = model(im)
    print(out["logits"].shape)
    print(out["preds"].shape)
