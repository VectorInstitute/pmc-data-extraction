"""Copy of CustomTextCLIP model from open_clip library.

Source: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L318
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.0  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = "learnable"
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "tok"
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        "avg"  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        "linear"  # linear projection for timm model output ('linear', 'mlp', '')
    )
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.0  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = "argmax"
    proj_bias: bool = False
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = "mlp"
    hf_pooler_type: str = "mean_pooler"  # attentional pooling for HF models


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout
            if vision_cfg.patch_dropout > 0
            else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(
            unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats
        )

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        image_features = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_features = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp(),
            }
            if self.logit_bias is not None:
                out_dict["logit_bias"] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return (
                image_features,
                text_features,
                self.logit_scale.exp(),
                self.logit_bias,
            )
        return image_features, text_features, self.logit_scale.exp()
