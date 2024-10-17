"""Wrapper for BiomedCLIP model loaded via open_clip library."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from open_clip.model import CustomTextCLIP
from torch import nn


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)
class BiomedCLIPText(nn.Module):
    """Wrapper around the `BiomedCLIP` model loaded via open_clip.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or a local path from which to load the model.
    pretrained : bool, default=True
        Whether to load the pretrained weights or not.
    use_all_token_embeddings : bool, default=False
        Whether to use all token embeddings for the text. If `False` the first token
        embedding will be used.
    freeze_layers : int | float | List[int] | bool, default=False
        Whether to freeze layers of the model and which layers to freeze. If `True`,
        all model layers are frozen. If it is an integer, the first `N` layers of
        the model are frozen. If it is a float, the first `N` percent of the layers
        are frozen. If it is a list of integers, the layers at the indices in the
        list are frozen.
    freeze_layer_norm : bool, default=True
        Whether to freeze the layer normalization layers of the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        use_all_token_embeddings: bool = False,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        # load model configs
        config_path = hf_hub_download(
            model_name_or_path, "open_clip_config.json", cache_dir=None
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_cfg = config["model_cfg"]

        # create model
        if model_config_kwargs is None:
            model_config_kwargs = {}
        model = CustomTextCLIP(**model_cfg, **model_config_kwargs)

        # load checkpoint file
        if pretrained:
            cached_file = hf_hub_download(
                model_name_or_path,
                "open_clip_pytorch_model.bin",
                revision=None,
                cache_dir=None,
            )
            self._load_checkpoint(model, cached_file)

        self.model = model.text

        # TODO: Does BiomedCLIP use normalize here or not?
        self.normalize = False
        self.emb_dim = 512

    def _load_checkpoint(
        self,
        model: CustomTextCLIP,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Any:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Certain text transformers no longer expect position_ids
        # after transformers==4.31
        position_id_key = "text.transformer.embeddings.position_ids"
        if position_id_key in state_dict and not hasattr(model, position_id_key):
            del state_dict[position_id_key]

        # Finally, load the massaged state_dict into model
        return model.load_state_dict(state_dict, strict=strict)

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The `input_ids` will be expected under the `Modalities.TEXT.name`
            key.

        Returns
        -------
        Tuple[torch.Tensor]
            The text embeddings. Will be a tuple with a single element.
        """
        input_ids = inputs[Modalities.TEXT.name]

        features = self.model(input_ids)
        features = F.normalize(features, dim=-1) if self.normalize else features

        return (features,)


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
    model_name_or_path="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)
class BiomedCLIPVision(nn.Module):
    """Wrapper around the `BiomedCLIP` model loaded via open_clip.

    Parameters
    ----------
    model_name_or_path : str
        The huggingface model name or a local path from which to load the model.
    pretrained : bool, default=True
        Whether to load the pretrained weights or not.
    use_all_token_embeddings : bool, default=False
        Whether to use all token embeddings for the text. If `False` the first token
        embedding will be used.
    freeze_layers : int | float | List[int] | bool, default=False
        Whether to freeze layers of the model and which layers to freeze. If `True`,
        all model layers are frozen. If it is an integer, the first `N` layers of
        the model are frozen. If it is a float, the first `N` percent of the layers
        are frozen. If it is a list of integers, the layers at the indices in the
        list are frozen.
    freeze_layer_norm : bool, default=True
        Whether to freeze the layer normalization layers of the model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        pretrained: bool = True,
        use_all_token_embeddings: bool = False,
        freeze_layers: Union[int, float, List[int], bool] = False,
        freeze_layer_norm: bool = True,
        model_config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        # load model configs
        config_path = hf_hub_download(
            model_name_or_path, "open_clip_config.json", cache_dir=None
        )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        model_cfg = config["model_cfg"]

        # create model
        if model_config_kwargs is None:
            model_config_kwargs = {}
        model = CustomTextCLIP(**model_cfg, **model_config_kwargs)

        # load checkpoint file
        if pretrained:
            cached_file = hf_hub_download(
                model_name_or_path,
                "open_clip_pytorch_model.bin",
                revision=None,
                cache_dir=None,
            )
            self._load_checkpoint(model, cached_file)

        self.model = model.visual

        # TODO: Does BiomedCLIP use normalize here or not?
        self.normalize = False
        self.emb_dim = 512

    def _load_checkpoint(
        self,
        model: CustomTextCLIP,
        checkpoint_path: str,
        strict: bool = True,
    ) -> Any:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Certain text transformers no longer expect position_ids
        # after transformers==4.31
        position_id_key = "text.transformer.embeddings.position_ids"
        if position_id_key in state_dict and not hasattr(model, position_id_key):
            del state_dict[position_id_key]

        # Finally, load the massaged state_dict into model
        return model.load_state_dict(state_dict, strict=strict)

    def forward(self, inputs: Dict[Union[str, Modality], Any]) -> Tuple[torch.Tensor]:
        """Run the forward pass.

        Parameters
        ----------
        inputs : Dict[str | Modality, Any]
            The input data. The image tensor will be expected under the `Modalities.RGB.name`
            key.

        Returns
        -------
        Tuple[torch.Tensor]
            The image embeddings. Will be a tuple with a single element.
        """
        input_ids = inputs[Modalities.RGB.name]

        features = self.model(input_ids)
        features = F.normalize(features, dim=-1) if self.normalize else features

        return (features,)
