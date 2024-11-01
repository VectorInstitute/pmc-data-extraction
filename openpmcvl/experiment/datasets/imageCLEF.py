import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop
from PIL import Image
from typing import Callable, Optional

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core.example import Example
from mmlearn.datasets.core import Modalities

# @external_store(group="datasets", root_dir=os.getenv("IMAGECLEF_ROOT_DIR"))
@external_store(group="datasets", root_dir=os.getenv("IMAGECLEF_2_ROOT_DIR"))
class ImageCLEF(Dataset[Example]):
    """ImageCLEF dataset for medical imaging modalities.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing 'train', 'val', 'test' directories.
    split : str
        Which dataset split to use ('train', 'val', 'test').
    transform : Optional[Callable], default=None
        Transform applied to the images.
    """

    def __init__(self, root_dir: str, split: str = 'train', transform: Optional[Callable[[Image.Image], torch.Tensor]] = None) -> None:
        """Initialize the ImageCLEF dataset."""
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform or Compose([
            Resize(224),
            CenterCrop(224),
            ToTensor()
        ])  # Default transform if none provided
        self.classes = sorted(os.listdir(self.root_dir))  # Assuming directories are the labels
        self.files = []
        self.labels = []

        # Gather all files and their corresponding labels
        for label_idx, modality in enumerate(self.classes):
            modality_path = os.path.join(self.root_dir, modality)
            for file_name in os.listdir(modality_path):
                if file_name.endswith('.jpg'):
                    self.files.append(os.path.join(modality_path, file_name))
                    self.labels.append(label_idx)

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        file_path = self.files[idx]
        label = self.labels[idx]

        # Open image file as a PIL Image
        image = Image.open(file_path).convert("RGB")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        # Return the data sample as an Example instance
        return Example({
            Modalities.RGB.name: image,
            Modalities.RGB.target: label,
            EXAMPLE_INDEX_KEY: idx,
            "image_path": file_path
        })

    @property
    def id2label(self) -> dict:
        """Return the label mapping."""
        return {idx: name for idx, name in enumerate(self.classes)}
    
    @property
    def zero_shot_prompt_templates(self) -> list:
        """Return simplified prompt templates for medical modality classification."""
        return [
            "{} scan image.",
            "{} medical image.",
            "{} diagnostic scan.",
            "Image from a {}.",
            "Diagnostic image: {}.",
            "Medical {} scan.",
            "Clinically used {} image.",
            "{} scan for diagnosis.",
            "Hospital {} image.",
            "Healthcare {} imaging."
        ]
