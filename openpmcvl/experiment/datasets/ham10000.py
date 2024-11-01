import os
from typing import Callable, Dict, Optional

import pandas as pd
import torch
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split

from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example


@external_store(group="datasets", root_dir=os.getenv("HAM10000_ROOT_DIR", MISSING))
class HAM10000(Dataset[Example]):
    """HAM10000 dataset for zero-shot classification.

    Parameters
    ----------
    root_dir : str
        Path to the dataset directory containing images and metadata CSV.
    transform : Optional[Callable], default=None
        Transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "test",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the HAM10000 dataset."""
        self.root_dir = root_dir
        
        # Check if the split-specific CSV files exist
        train_csv = os.path.join(root_dir, "HAM10000_train.csv")
        test_csv = os.path.join(root_dir, "HAM10000_test.csv")

        if not os.path.exists(train_csv) or not os.path.exists(test_csv):
            # Load the original metadata CSV
            original_metadata = pd.read_csv(os.path.join(root_dir, "HAM10000_metadata.csv"))
            # Split the data into train and test
            train_data, test_data = train_test_split(original_metadata, test_size=0.2, random_state=42)
            # Save the splits as new CSV files
            train_data.to_csv(train_csv, index=False)
            test_data.to_csv(test_csv, index=False)

        # Load the metadata for the requested split
        if split == 'train':
            self.metadata = pd.read_csv(train_csv)
        elif split == 'test':
            self.metadata = pd.read_csv(test_csv)
        else:
            raise ValueError("Split must be 'train' or 'test'")
        
        self.classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]

        self.transform = (
            Compose([Resize(224), CenterCrop(224), ToTensor()])
            if transform is None
            else transform
        )

    @property
    def zero_shot_prompt_templates(self) -> list[str]:
        """Return the zero-shot prompt templates."""
        return [
            "a histopathology slide showing {}",
            "histopathology image of {}",
            "pathology tissue showing {}",
            "presence of {} tissue on image",
        ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample as an Example instance."""
        entry = self.metadata.iloc[idx]
        image_path = os.path.join(
            self.root_dir, "skin_cancer", f"{entry['image_id']}.jpg"
        )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label_index = self.classes.index(entry["dx"])

        return Example(
            {
                Modalities.RGB.name: image,
                Modalities.RGB.target: label_index,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the label mapping."""
        return {
            0: "Melanocytic Nevi",
            1: "Melanoma",
            2: "Benign Keratosis-like Lesions",
            3: "Basal Cell Carcinoma",
            4: "Actinic Keratoses and Intraepithelial Carcinoma",
            5: "Vascular Lesions",
            6: "Dermatofibroma",
        }
        
    @property
    def is_multilabel(self):
        return False
