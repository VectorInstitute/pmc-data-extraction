"""Wrapper for PubmedCLIP vision and text encoders.

PubmedCLIP[1] provides three checkpoints: ResNet-50, ResNet-50x4 and ViT32.
This implementation is ViT32 only.

References
----------
[1] https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32
"""

from typing import Any, Dict, List, Tuple, Union

import torch
from mmlearn.conf import external_store
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from PIL import Image
from torch import nn
from transformers import CLIPModel, CLIPProcessor


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
)
class PubmedClipVision(nn.Module):
    """Wrapper for vision encoder of PubmedCLIP."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load the whole model
        model = CLIPModel.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )
        self.model = model

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
        image_embeds = self.model.get_image_features(input_ids)
        return (image_embeds,)


@external_store(
    group="modules/encoders",
    provider="openpmcvl",
)
class PubmedClipText(nn.Module):
    """Wrapper for text encoder of PubmedCLIP."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load the whole model
        model = CLIPModel.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )
        self.model = model

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
        attention_mask = inputs["attention_mask"]
        text_embeds = self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return (text_embeds,)


@external_store(group="datasets/tokenizers", provider="openpmcvl")
class PubmedClipTokenizer:
    """Wrapper for PubmedCLIP's tokenizer."""

    def __init__(self, context_length: int = 77) -> None:
        """Initialize the model."""
        super().__init__()

        # load via open_clip
        self.processor = CLIPProcessor.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )

        self.context_length = context_length

    def __call__(self, x: Union[str, List[str]]) -> Any:
        """Pass any input to loaded tokenizer."""
        inputs = self.processor(text=x, images=None, return_tensors="pt", padding=True)
        return {
            Modalities.TEXT.name: inputs["input_ids"][:, : self.context_length],
            "attention_mask": inputs["attention_mask"][:, : self.context_length],
        }


@external_store(group="datasets/transforms", provider="openpmcvl")
class PubmedClipTransform:
    """Wrapper for PubmedCLIP's transforms."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load via open_clip
        self.processor = CLIPProcessor.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Pass any input to loaded transform."""
        inputs = self.processor(
            text=None, images=image, return_tensors="pt", padding=True
        )
        pixel_values = inputs["pixel_values"].squeeze()
        assert isinstance(pixel_values, torch.Tensor)
        return pixel_values
