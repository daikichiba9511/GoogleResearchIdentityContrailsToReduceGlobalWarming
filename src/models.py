import itertools
from logging import getLogger
from typing import Any, Sequence, TypeVar

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unetr import UNETR
from segmentation_models_pytorch.base import initialization as smp_init
from segmentation_models_pytorch.base import modules as md
from transformers import (
    OneFormerModel,
    OneFormerProcessor,
    SegformerForSemanticSegmentation,
)

logger = getLogger(__name__)


# TODO: まだ動かせてない
# ImageProcessorをDataset側に持たせた方が良いかも
# Ref:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormer_on_a_panoptic_dataset.ipynb
class OneFormerForwarder(nn.Module):
    def __init__(self):
        super().__init__()
        # file = "shi-labs/oneformer_ade20k_dinat_large"
        file = "shi-labs/oneformer_ade20k_swin_tiny"
        self.model = OneFormerModel.from_pretrained(file)  # type: ignore
        # self.processor = OneFormerImageProcessor(
        #     do_resize=False,
        #     do_normalize=False,
        #     do_rescale=False,
        #     repo_path="shi-labs/oneformer_ade20k_swin_tiny",
        # )
        self.processor = OneFormerProcessor.from_pretrained(file)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(self.model, OneFormerModel)
            and self.processor is not None
            and isinstance(self.processor, OneFormerProcessor)
        ):
            # x: (B, C, H, W)
            # ここがうまくいかない
            # x = (x - x.mean()) / (x.std() + 1e-8)
            x = (x - x.min()) / ((x.max() - x.min()) + 1e-8)
            input = self.processor(x, ["semantic"], return_tensors="pt")
            output = self.model(**input)
            print(output)
            return output.transformer_decoder_mask_predictions

        raise NotImplementedError


_T = TypeVar("_T", bound=torch.Tensor)


class TTA:
    _config = {"d8prob": (8, True)}

    def __init__(self, tta_type: str) -> None:
        self.n_times, self.prob = self._config[tta_type]

    @staticmethod
    def _tta_stack(x: torch.Tensor, n_times: int) -> torch.Tensor:
        """Increse input x by n_times TTA patterns
        batch_size = n * n_times
        """

        def _augmentated(k: int) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            if n_times == 8:
                rotated = torch.rot90(x, k=k, dims=(2, 3))
                flipped = torch.flip(rotated, dims=(3,))
                return rotated, flipped
            return torch.rot90(x, k=k, dims=(2, 3))

        augmented_x = list(map(_augmentated, range(4)))
        flattend_x = list(itertools.chain(*augmented_x))
        flattend_x = torch.cat(flattend_x, dim=0)
        return flattend_x

    @staticmethod
    def _tta_average(preds: torch.Tensor, n_times: int, prob: bool) -> torch.Tensor:
        batch_size, channels, height, width = preds.shape
        y_preds = preds.view(n_times, batch_size // n_times, channels, height, width)
        _batch_size = batch_size // n_times
        y_avg = torch.zeros(
            (_batch_size, 1, height, width), dtype=torch.float32, device=preds.device
        )
        if prob:
            y_preds = torch.sigmoid(y_preds)

        if n_times == 4:
            for k in range(4):
                y_avg += (1 / n_times) * torch.rot90(y_preds[k], k=-k, dims=(2, 3))

        if n_times == 8:
            for k in range(4):
                y_avg += (1 / n_times) * torch.rot90(y_preds[2 * k], k=-k, dims=(2, 3))
                flipped = torch.flip(y_preds[2 * k + 1], dims=(3,))
                y_avg += (1 / n_times) * torch.rot90(flipped, k=-k, dims=(2, 3))

        if prob:
            return y_avg.clamp(min=1e-6, max=1 - 1e-6).logit()
        return y_avg

    def average(self, y: torch.Tensor) -> torch.Tensor:
        """Average TTA predictions"""
        if self.n_times == 1:
            return y
        return self._tta_average(y, self.n_times, self.prob)

    def stack(self, y: torch.Tensor) -> torch.Tensor:
        """Stack TTA predictions"""
        if self.n_times == 1:
            return y
        return self._tta_stack(y, self.n_times)


class DecoderBlock(nn.Module):
    """U-Net decoder from Segmentation Models PyTorch

    Ref:
    - https://github.com/qubvel/segmentation_models.pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        conv_in_channels = in_channels + skip_channels
        self.conv1 = md.Conv2dReLU(
            in_channels=conv_in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = md.Conv2dReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.dropout_skip = nn.Dropout(p=dropout)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            skipped = self.dropout_skip(skip)
            x = torch.cat([x, skipped], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        use_batchnorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder_channels = encoder_channels[::-1]
        # -- computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels
        self.center = nn.Identity()

        # -- Combine decoder keyword arguments
        def _decorder_block(
            channels: tuple[int, int, int],
        ) -> DecoderBlock:
            in_ch, skip_ch, out_ch = channels
            return DecoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                skip_channels=skip_ch,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )

        self.blocks = nn.ModuleList(
            map(_decorder_block, zip(in_channels, skip_channels, out_channels))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features[::-1]  # reverse channels to start from head of encoder
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        # TODO: ここもっとわかりやすく書けるはず
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


def _check_reduction(reduction_factors: Sequence[int]) -> None:
    r_prev = 1
    for r in reduction_factors:
        if r / r_prev != 2:
            raise ValueError(
                "Reduction factor of each block should be divisible by the previous one"
            )
        r_prev = r


class CustomedUnet(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        decoder_channels: list[int] = [256, 128, 64, 32, 16],
        dropout: float = 0.0,
        tta_type: str | None = None,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            name, features_only=True, pretrained=pretrained
        )
        encoder_channels = self.encoder.feature_info.channels()
        _check_reduction(self.encoder.feature_info.reduction())
        if len(encoder_channels) != len(decoder_channels):
            raise ValueError(
                "Encoder channels and decoder channels should have the same length"
            )
        self.decoder = UnetDecoder(encoder_channels, decoder_channels, dropout=dropout)
        self.segmentation_head = smp.base.SegmentationHead(
            decoder_channels[-1], out_channels=1, activation=None, kernel_size=1
        )
        smp_init.initialize_decoder(self.decoder)
        self.asym_conv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=5, stride=2, padding=2),  # downsample 2x
            nn.ReLU(inplace=True),
            nn.Conv2d(25, 1, kernel_size=1),
        )
        self.tta = TTA(tta_type) if tta_type is not None else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        if self.tta is not None:
            x = self.tta.stack(x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)
        logits = self.segmentation_head(decoder_output)

        if self.tta is not None:
            logits = self.tta.average(logits)

        preds = self.asym_conv(logits)
        return {"logits": logits, "preds": preds, "cls_logits": None}


class ContrailsModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_weight: str | None = None,
        aux_params: dict[str, Any] | None = None,
        arch: str = "Unet",
        trainable_downsampling: bool = True,
    ) -> None:
        super().__init__()
        if (
            encoder_name.startswith("tu-convnext")
            # or encoder_name.startswith("tu-maxvit")
            or encoder_name.startswith("tu-pvt")
        ):
            encoder_depth = 4
        else:
            encoder_depth = 5

        pop_last_block = False
        if encoder_name.startswith("tu-convnext") or encoder_name.startswith("tu-pvt"):
            decoder_channels = [256, 128, 64, 32]
            # decoder_channels = [128, 64, 32, 16]
            # decoder_channels = [512, 256, 128, 64]

        elif encoder_name.startswith("tu-maxvit"):
            # NOTE:
            # docoderの最後のブロックをpopして64->32の変換からSegmentationHeadに渡してる
            # ので、Headは64で初期化するのに64 -> 64にする
            # SegmentationHeadはchannel方向の集約のみなのでOK
            # decoder_channels = [256, 128, 64, 64]
            # pop_last_block = False
            decoder_channels = [256, 128, 64, 32, 16]
        else:
            decoder_channels = [256, 128, 64, 32, 16]

        if arch == "Unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weight,
                encoder_depth=encoder_depth,
                decoder_channels=decoder_channels,
                decoder_use_batchnorm=True,
                in_channels=3,
                classes=1,
                activation=None,
                aux_params=aux_params,
            )
            if pop_last_block:
                self.model.decoder.blocks.pop(-1)

        elif arch == "SwinUNETR":
            self.model = SwinUNETR(
                img_size=(512, 512),
                in_channels=3,
                out_channels=1,
                spatial_dims=2,
                use_v2=False,
                use_checkpoint=True,
            )

        elif arch == "OneFormer":
            self.model = OneFormerForwarder()
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

        if trainable_downsampling:
            # NOTE:
            # IN : torch.nn.Conv2d(1,1,kernel_size=(5,5),stride=2,padding=2)(torch.randn(3,1,512,512)).shape
            # OUT: torch.Size([3, 1, 256, 256])
            # padding=2で微妙に足りない分を補う
            thin_downsampling = False
            if thin_downsampling:
                self.downsample2x = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=2)
            else:
                hidden_size = 25
                self.downsample2x = nn.Sequential(
                    nn.Conv2d(1, hidden_size, kernel_size=5, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_size, 1, kernel_size=1),
                )
        else:
            # self.downsample2x = nn.AvgPool2d(kernel_size=2, stride=2)
            self.downsample2x = tv.transforms.Resize(
                size=(256, 256),
                interpolation=tv.transforms.InterpolationMode.BILINEAR,
                max_size=256,
                antialias=True,  # type: ignore
            )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(images)
        if isinstance(outputs, tuple):
            logits, cls_logits = outputs
            cls_logits = cls_logits.reshape(-1)
        else:
            logits = outputs
            cls_logits = None

        if logits.shape[-1] != 256:
            # preds = nn.functional.interpolate(
            #     # logits, size=(256, 256), mode="bicubic", align_corners=False
            #     logits,
            #     size=(256, 256),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            preds = self.downsample2x(logits)
        else:
            preds = logits
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

    # model = ContrailsModel(encoder_name="swinv2", arch="SwinUNETR")
    # im = torch.randn(8, 3, 512, 512)
    # out = model(im)
    # print(out["logits"].shape)
    # print(out["preds"].shape)

    model = ContrailsModel(encoder_name="tu-maxvit_small_tf_512", arch="Unet")
    # model = torch.compile(model)
    im = torch.randn(8, 3, 512, 512)
    out = model(im)
    print(out["logits"].shape)
    print(out["preds"].shape)
    #
    # model = ContrailsModel(encoder_name="tu-convnext_small", arch="Unet")
    # # model = torch.compile(model)
    # im = torch.randn(8, 3, 512, 512)
    # out = model(im)
    # print(out["logits"].shape)
    # print(out["preds"].shape)

    # model = ContrailsModel(encoder_name="swin", arch="OneFormer")
    # # model = torch.compile(model)
    # im = torch.randn(8, 3, 512, 512)
    # out = model(im)
    # print(out["logits"].shape)
    # print(out["preds"].shape)

    model = ContrailsModel(encoder_name="tu-pvt_v2_b1", arch="Unet")
    # model = torch.compile(model)
    im = torch.randn(8, 3, 512, 512)
    out = model(im)
    print(out["logits"].shape)
    print(out["preds"].shape)

    model = ContrailsModel(encoder_name="tu-tf_efficientnetv2_s", arch="Unet")
    # model = torch.compile(model)
    im = torch.randn(8, 3, 512, 512)
    out = model(im)
    print(out["logits"].shape)
    print(out["preds"].shape)

    model = CustomedUnet(name="maxvit_small_tf_512", pretrained=True)
    im = torch.randn(8, 3, 512, 512)
    out = model(im)
    print(out["logits"].shape)
    print(out["preds"].shape)
