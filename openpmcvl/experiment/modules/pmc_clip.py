"""The ResNet architecture used in PMC-CLIP.

Extracted from a tutorial notebook[1] referenced on PMC-CLIP's GitHub repo[2].

References
----------
[1] https://colab.research.google.com/drive/1P7uyzK_Mhu1YyMeRrrRY_e3NpkNBOI4L?usp=sharing#scrollTo=ERh21bj8wHWH
[2] https://github.com/WeixiongLin/PMC-CLIP
"""

import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from torch import nn
from torchvision import transforms
from transformers import AutoModel


class Bottleneck(nn.Module):
    """Define image tower."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1):  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after
        # the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool
            # and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(
                OrderedDict(
                    [
                        ("-1", nn.AvgPool2d(stride)),
                        (
                            "0",
                            nn.Conv2d(
                                inplanes,
                                planes * self.expansion,
                                1,
                                stride=1,
                                bias=False,
                            ),
                        ),
                        ("1", nn.BatchNorm2d(planes * self.expansion)),
                    ]
                )
            )

    def forward(self, x: torch.Tensor):  # type: ignore[no-untyped-def]
        """Compute features."""
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu3(out)


class AttentionPool2d(nn.Module):
    """Attention Pool for ResNet."""

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads: int,
        output_dim: Optional[int] = None,
    ):
        """Initialize the module."""
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):  # type: ignore[no-untyped-def]
        """Compute features."""
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A modified version of torchvision's ResNet class.

    The modifications are listed below:
        - There are now 3 "stem" convolutions as opposed to 1, with an average
        pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is
        prepended to convolutions with stride > 1
        - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):  # type: ignore[no-untyped-def]
        """Initialize the module."""
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(
            3, width // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width // 2, width // 2, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])  # type: ignore[no-untyped-call]
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # type: ignore[no-untyped-call]
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # type: ignore[no-untyped-call]
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # type: ignore[no-untyped-call]

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()  # type: ignore[no-untyped-call]

    def _make_layer(self, planes, blocks, stride=1):  # type: ignore[no-untyped-def]
        """Make layer."""
        layers = [Bottleneck(self._inplanes, planes, stride)]  # type: ignore[no-untyped-call]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))  # type: ignore[no-untyped-call]

        return nn.Sequential(*layers)

    def init_parameters(self):  # type: ignore[no-untyped-def]
        """Initialize parameters."""
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features**-0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):  # type: ignore[no-untyped-def]
        """Freeze part or all of the parameters."""
        assert (
            unlocked_groups == 0
        ), "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            # TODO: freeze_batch_norm_2d(self) but freeze_batch_norm_2d isn't defined
            pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):  # type: ignore[no-untyped-def]
        """Checkpoint gradients."""
        # FIXME support for non-transformer
        pass

    def stem(self, x):  # type: ignore[no-untyped-def]
        """Stem of ResNet model."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return self.avgpool(x)

    def forward(self, x):  # type: ignore[no-untyped-def]
        """Encode inputs."""
        x = self.stem(x)  # type: ignore[no-untyped-call]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        visual_output = dict.fromkeys(["image_features", "mim_loss"], None)
        visual_output.update(
            {
                "image_features": x,
            }
        )

        return visual_output


@external_store(group="datasets/transforms")
def pmc_clip_vision_transform(image_crop_size: int = 224) -> transforms.Compose:
    """Return transforms for training/evaluating PMC-CLIP with medical images.

    Parameters
    ----------
    image_crop_size : int, default=224
        Size of the image crop.

    Returns
    -------
    transforms.Compose
        Composed transforms for training CLIP with medical images.
    """
    if (
        isinstance(image_crop_size, (list, tuple))
        and image_crop_size[0] == image_crop_size[1]
    ):
        # for square size, pass size as int so that Resize() uses aspect preserving
        # shortest edge
        image_crop_size = image_crop_size[0]

    return transforms.Compose(
        [
            transforms.Resize(
                image_crop_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
    pretrained=True,
    ckpt_dir=os.getenv("PMC_CLIP_ROOT", ""),
)
class PmcClipVision(nn.Module):
    """Wrapper for vision encoder of PMC-CLIP."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_dir: str = "",
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        pretrained: bool, default=True
            Whether to load PMC-CLIP's weights pretrained on PMC-OA.
        ckpt_dir: str, default=""
            If pretrained is True, must be set to the directory
            where three checkpoints from PMC-CLIP are stored:
            "image_encoder_resnet50.pth",
            "text_encoder.pth",
            "text_projection_layer.pth"
        """
        super().__init__()

        # load image encoder
        image_encoder = ModifiedResNet(  # type: ignore[no-untyped-call]
            layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64
        )

        # load pretrained weights
        if pretrained:
            image_encoder.load_state_dict(
                torch.load(os.path.join(ckpt_dir, "image_encoder_resnet50.pth"))
            )

        self.image_encoder = image_encoder

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The image tensor will be expected under the
            `Modalities.RGB.name` key.

        Returns
        -------
        Tuple[torch.Tensor]
            The image embeddings. Will be a tuple with a single element.
        """
        input_ids = inputs[Modalities.RGB.name]

        features = self.image_encoder(input_ids)
        if isinstance(features, dict):
            features = features["image_features"]

        return (features,)


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
    pretrained=True,
    ckpt_dir=os.getenv("PMC_CLIP_ROOT", ""),
)
class PmcClipText(nn.Module):
    """Wrapper for text encoder of PMC-CLIP."""

    def __init__(
        self,
        pretrained: bool = True,
        ckpt_dir: str = "",
        modality: str = "text",
    ) -> None:
        """Initialize the model.

        Parameters
        ----------
        pretrained: bool, default=True
            Whether to load PMC-CLIP's weights pretrained on PMC-OA.
        ckpt_dir: str, default=""
            If pretrained is True, must be set to the directory
            where three checkpoints from PMC-CLIP are stored:
            "image_encoder_resnet50.pth",
            "text_encoder.pth",
            "text_projection_layer.pth"
        modality: str, default="text"
            The modality to encode.
        """
        super().__init__()

        # load text encoder
        text_encoder = AutoModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        )
        # load pretrained weights
        if pretrained:
            text_encoder.load_state_dict(
                torch.load(os.path.join(ckpt_dir, "text_encoder.pth"))
            )

        # load text proj layer
        text_projection_layer = torch.load(
            os.path.join(ckpt_dir, "text_projection_layer.pth")
        )
        if not pretrained:
            text_projection_layer = torch.randn_like(text_projection_layer)
        text_projection_layer = nn.Parameter(text_projection_layer)

        self.text_encoder = text_encoder
        self.text_projection_layer = text_projection_layer
        self.modality = modality

    def forward(
        self, inputs: Dict[Union[str, Modality], Any]
    ) -> Union[Tuple[torch.Tensor], Dict[str, torch.Tensor]]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The `input_ids` will be expected under the
            `Modalities.TEXT.name` key.
            If self.modality is set to "patient", then keys
            `Modalities.PATIENT_Q.name` and `Modalities.PATIENT_T.name`
            are expected.

        Returns
        -------
        Tuple[torch.Tensor]
            The image embeddings. Will be a tuple with a single element.
        """
        if self.modality == "patient":
            input_ids_q = inputs[Modalities.PATIENT_Q.name]
            input_ids_t = inputs[Modalities.PATIENT_T.name]

            features_q = self.text_encoder(input_ids_q)
            features_t = self.text_encoder(input_ids_t)
            pooler_output_q = features_q.pooler_output
            pooler_output_t = features_t.pooler_output
            features_q = pooler_output_q @ self.text_projection_layer
            features_t = pooler_output_t @ self.text_projection_layer

            return {
                Modalities.PATIENT_Q.name: features_q,
                Modalities.PATIENT_T.name: features_t,
            }
        # general input
        input_ids = inputs[self.modality]

        features = self.text_encoder(input_ids)
        pooler_output = features.pooler_output
        features = pooler_output @ self.text_projection_layer

        return (features,)
