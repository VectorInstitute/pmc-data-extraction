"""The ResNet architecture used in PMC-CLIP.

Extracted from a tutorial notebook[1] referenced on PMC-CLIP's GitHub repo[2].

References
----------
[1] https://colab.research.google.com/drive/1P7uyzK_Mhu1YyMeRrrRY_e3NpkNBOI4L?usp=sharing#scrollTo=ERh21bj8wHWH
[2] https://colab.research.google.com/drive/1P7uyzK_Mhu1YyMeRrrRY_e3NpkNBOI4L?usp=sharing#scrollTo=ERh21bj8wHWH
"""

from collections import OrderedDict
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision import transforms

from transformers import AutoTokenizer, AutoModel

from PIL import Image
import os

from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from typing import Any, Dict, List, Optional, Tuple, Union


class Bottleneck(nn.Module):
    """Define image tower"""
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """Initialize the module."""
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
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
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        """Compute features."""
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """Attention Pool for ResNet."""

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """Initialize the module."""
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        """Compute features."""
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains several changes.

    Changes are listed below:
        - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
        - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
        - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        """Initialize the module."""
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        """Make layer."""
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        """Initialize parameters."""
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """Freeze part or all of the parameters."""
        assert unlocked_groups == 0, "partial locking not currently supported for this model"
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        """Stem of ResNet model."""
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        """Encoder inputs."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        visual_output = dict.fromkeys(["image_features", "mim_loss"], None)
        visual_output.update({
            "image_features": x,
        })

        return visual_output


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
    if isinstance(image_crop_size, (list, tuple)) and image_crop_size[0] == image_crop_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_crop_size = image_crop_size[0]

    return transforms.Compose([
        transforms.Resize(image_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711]),
    ])


class PmcClipVision(nn.Module):
    """Wrapper for vision encoder of PMC-CLIP."""
    def __init__(self,
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
        image_encoder = ModifiedResNet(layers=[3, 4, 6, 3], output_dim=768, heads=8, image_size=224, width=64)

        # load pretrained weights
        if pretrained:
            image_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "image_encoder_resnet50.pth")))

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


class PmcClipText(nn.Module):
    """Wrapper for text encoder of PMC-CLIP."""
    def __init__(self,
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

        # load text encoder
        text_encoder = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        # load pretrained weights
        if pretrained:
            text_encoder.load_state_dict(torch.load(os.path.join(ckpt_dir, "text_encoder.pth")))

        # load text proj layer
        text_projection_layer = torch.load(os.path.join(ckpt_dir, "text_projection_layer.pth"))
        if not pretrained:
            text_projection_layer = torch.randn_like(text_projection_layer)
        text_projection_layer = nn.Parameter(text_projection_layer)

        self.text_encoder = text_encoder
        self.text_projection_layer = text_projection_layer

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
        input_ids = inputs[Modalities.TEXT.name]

        features = self.text_encoder(input_ids)
        last_hidden_state = features.last_hidden_state
        pooler_output = features.pooler_output
        features = pooler_output @ self.text_projection_layer

        return (features,)

if __name__ == "__main__":
    import os
    pmc_clip_root = os.getenv("PMC_CLIP_ROOT", "")

    # Load Image Encoder
    image_encoder = PmcClipVision(pretrained=True, ckpt_dir=pmc_clip_root)

    # Load Text Encoder
    text_encoder = PmcClipText(pretrained=True, ckpt_dir=pmc_clip_root)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

    # Logit Scale
    logit_scale = 4.4292

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_encoder = image_encoder.to(device)
    text_encoder = text_encoder.to(device)
    # text_projection_layer = text_projection_layer.to(device)

    # Load Image
    preprocess_val = pmc_clip_vision_transform(
        image_crop_size=224,
    )

    image_path_ls = [
        '/h/yaspar/Documents/GitHub/pmc-data-extraction-dev/openpmcvl/experiment/figures/chest_X-ray.jpg',
        '/h/yaspar/Documents/GitHub/pmc-data-extraction-dev/openpmcvl/experiment/figures/brain_MRI.jpg'
    ]
    images = []
    image_tensor = []
    for image_path in image_path_ls:
        image = Image.open(image_path).convert('RGB')
        images.append(image)
        image_tensor.append(preprocess_val(image))

    image_tensor = torch.stack(image_tensor, dim=0).to(device)

    # Extract Image feature
    inputs = {"rgb": image_tensor}
    image_feature = image_encoder(inputs)
    if isinstance(image_feature, dict):
        image_feature = image_feature['image_features']
    image_feature = image_feature[0]
    print(f"image_feature.shape={image_feature.shape}")

    # Load Text
    bert_input = [
        'chest X-ray',
        'brain MRI',
    ]
    encoded_input = tokenizer(bert_input, padding='max_length', truncation=True, max_length=77, return_tensors='pt')
    input_ids = encoded_input['input_ids'].to(device)
    print(f"text_input_ids.shape: {input_ids.shape}")

    # Extract Text feature
    inputs = {"rgb": image_tensor, "text": input_ids}
    text_feature = text_encoder(inputs)
    text_feature = text_feature[0]
    print(f"text_feature.shape={text_feature.shape}")

    # Calculate Similarity
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
    similarity = (math.exp(logit_scale) * image_feature @ text_feature.T).softmax(dim=-1)

    for i, (image_path, image) in enumerate(zip(image_path_ls, images)):
        print(image_path)
        for j in range(len(bert_input)):
            print(f'{bert_input[j]}: {similarity[i, j].item()}')
        print('\n')