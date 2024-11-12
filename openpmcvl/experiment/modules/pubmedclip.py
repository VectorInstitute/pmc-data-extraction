"""Wrapper for PubmedCLIP vision and text encoders.

PubmedCLIP[1] provides three checkpoints: ResNet-50, ResNet-50x4 and ViT32.
This implementation is ViT32 only.

References
----------
[1] https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32
"""

import requests
from PIL import Image
import torch
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.modalities import Modality
from torch import nn
from mmlearn.datasets.core import Modalities
from torchvision import transforms

from transformers import CLIPProcessor, CLIPModel
from typing import Any, Dict, Optional, Tuple, Union, List


class PubmedClipVision(nn.Module):
    """Wrapper for vision encoder of PubmedCLIP."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load the whole model
        model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
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


class PubmedClipText(nn.Module):
    """Wrapper for text encoder of PubmedCLIP."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load the whole model
        model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
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
        text_embeds = self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return (text_embeds,)


class PubmedClipTokenizer:
    """Wrapper for PubmedCLIP's tokenizer."""

    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load via open_clip
        self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

    def __call__(self, x: Union[str, List[str]]) -> Any:
        """Pass any input to loaded tokenizer."""
        inputs = self.processor(text=x, images=None, return_tensors="pt", padding=True)
        return {Modalities.TEXT.name: inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]}


class PubmedClipTransform:
    """Wrapper for PubmedCLIP's transforms."""
    def __init__(self) -> None:
        """Initialize the model."""
        super().__init__()

        # load via open_clip
        self.processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

    def __call__(self, image: Image) -> torch.Tensor:
        """Pass any input to loaded transform."""
        inputs = self.processor(text=None, images=image, return_tensors="pt", padding=True)
        return inputs["pixel_values"].squeeze()


if __name__ == "__main__":
    model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")

    # print(model)
    # print(processor)

    url = "https://huggingface.co/flaviagiammarino/pubmed-clip-vit-base-patch32/resolve/main/scripts/input.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)
    text = ["Chest X-Ray", "Brain MRI", "Abdominal CT Scan"]

    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    probs = model(**inputs).logits_per_image.softmax(dim=1).squeeze()

    # new implementation
    tokenizer = PubmedClipTokenizer()
    tokens = tokenizer(text)
    transform = PubmedClipTransform()
    pixel_values = transform(image).unsqueeze(0)
    inputs_ = {"rgb": pixel_values}
    inputs_.update(tokens)
    image_encoder = PubmedClipVision()
    image_embeds = image_encoder(inputs_)
    image_embeds = image_embeds[0]
    print(image_embeds.shape)

    exit()

    text_encoder = PubmedClipText()
    text_embeds = text_encoder(inputs_)

    print(f"inputs: {inputs}")
    print(f"probs: {probs}")
    print(f"inputs.keys(): {inputs.keys()}")
    print(f"inputs.input_ids.shape: {inputs['input_ids'].shape}")
    print(f"inputs.attention_mask.shape: {inputs['attention_mask'].shape}")
    print(f"inputs.pixel_values.shape: {inputs['pixel_values'].shape}")
    print(f"probs.shape: {probs.shape}")

    inputs2 = processor(text=None, images=image, return_tensors="pt", padding=True)
    print(f"inputs: {inputs2}")
    print(f"inputs.keys(): {inputs2.keys()}")
    print(f"inputs.pixel_values.shape: {inputs2['pixel_values'].shape}")
    print(f"inputs.pixel_values.shape: {inputs2['pixel_values'].squeeze().shape}")
    print(torch.equal(inputs["pixel_values"], inputs2["pixel_values"]))

    inputs3 = processor(text=text, images=None, return_tensors="pt", padding=True)
    print(f"inputs: {inputs3}")
    print(f"inputs.keys(): {inputs3.keys()}")
    print(f"inputs.input_ids.shape: {inputs3['input_ids'].shape}")
    print(f"inputs.attention_mask.shape: {inputs3['attention_mask'].shape}")
    print(torch.equal(inputs["input_ids"], inputs3["input_ids"]))
    print(torch.equal(inputs["attention_mask"], inputs3["attention_mask"]))