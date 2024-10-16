"""MIMIC-IV-CXR Dataset."""

import json
import logging
import os
from typing import Callable, Literal, Optional, get_args

import numpy as np
import pandas as pd
import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from datasets import load_from_disk, load_dataset


logger = logging.getLogger(__name__)


@external_store(
    group="datasets",
    root_dir=os.getenv("LC25000_LUNG_ROOT_DIR", MISSING),
    organ="lung",
    split="train",
)
class LC25000(Dataset):  # type: ignore[type-arg]
    """Module to load images and labels from LC25000 dataset.

    Parameters
    ----------
    root_dir : str
        Path to the directory containing json files which describe data entries.
    organ: {"lung", "colon"}
        Determines the subset of LC25000 to be loaded.
    split : {"train", "test"}
        Dataset split.
    transform :  Optional[Callable]
        Custom transform applied to images.
    """

    def __init__(
        self,
        root_dir: str,
        organ: Literal["lung", "colon"],
        split: Literal["train", "test"],
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        """Initialize the dataset."""
        if os.path.exists(os.path.join(root_dir, f"cache/lc25000_{organ}_{split}.arrow")):
            print("!!!Using cached dataset")
            dataset = load_from_disk(os.path.join(root_dir, f"cache/lc25000_{organ}_{split}.arrow"))
        else:
            os.makedirs(os.path.join(root_dir, "cache/"), exist_ok=True)

            dataset = load_dataset(
                "1aurent/LC25000",
                cache_dir=os.path.join(root_dir, "scratch/"),
            )["train"]
            dataset = dataset.filter(lambda row: row["organ"] == organ)

            datasets_dict = dataset.train_test_split(test_size=0.2, shuffle=True,
                                                     train_indices_cache_file_name=os.path.join(root_dir, f"cache/lc25000_{organ}_train_indices.arrow"),
                                                     test_indices_cache_file_name=os.path.join(root_dir, f"cache/lc25000_{organ}_test_indices.arrow"))

            datasets_dict["train"].save_to_disk(os.path.join(root_dir, f"cache/lc25000_{organ}_train.arrow"))
            datasets_dict["test"].save_to_disk(os.path.join(root_dir, f"cache/lc25000_{organ}_test.arrow"))
        self.data = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = ToTensor()

        if organ == "lung":
            self.labels_text = ["benign lung", "lung adenocarcinoma", "lung squamous cell carcinoma"]
        elif organ == "colon":
            self.labels_text = ["benign colonic tissue", "colon adenocarcinoma"]

        self.templates = ["a histopathology slide showing {}",
                          "histopathology image of {}",
                          "pathology tissue showing {}",
                          "presence of {} tissue on image"]

    def __getitem__(self, idx: int) -> Example:
        """Return all the images and the label vector of the idx'th study."""
        image = self.data[idx]["image"]
        image = self.transform(image)
        label = int(self.data[idx]["label"])

        example = Example({Modalities.RGB: image,
                           Modalities.RGB.target: label,
                           EXAMPLE_INDEX_KEY: idx})

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)
