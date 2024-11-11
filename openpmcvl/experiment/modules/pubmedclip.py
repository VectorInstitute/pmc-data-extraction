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

from transformers import CLIPProcessor, CLIPModel
from typing import Any, Dict, Optional, Tuple, Union


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

        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = self.model.config.output_attentions
        output_hidden_states = self.model.config.output_hidden_states
        return_dict = self.model.config.use_return_dict

        vision_outputs = self.model.vision_model(
            pixel_values=input_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.model.visual_projection(image_embeds)

        return (image_embeds,)







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
    inputs_ = {"rgb": inputs["pixel_values"], "text_mask": inputs["attention_mask"], "text": inputs["input_ids"]}
    image_encoder = PubmedClipVision()
    image_embeds = image_encoder(inputs_)
    print(image_embeds)
    print(image_embeds[0].shape)
    exit()


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
    print(torch.equal(inputs["pixel_values"], inputs2["pixel_values"]))

    inputs3 = processor(text=text, images=None, return_tensors="pt", padding=True)
    print(f"inputs: {inputs3}")
    print(f"inputs.keys(): {inputs3.keys()}")
    print(f"inputs.input_ids.shape: {inputs3['input_ids'].shape}")
    print(f"inputs.attention_mask.shape: {inputs3['attention_mask'].shape}")
    print(torch.equal(inputs["input_ids"], inputs3["input_ids"]))
    print(torch.equal(inputs["attention_mask"], inputs3["attention_mask"]))