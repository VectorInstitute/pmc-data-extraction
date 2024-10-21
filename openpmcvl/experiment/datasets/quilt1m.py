"""Quilt-1M Dataset."""

import ast
import os
from typing import Callable, Dict, List, Literal, Optional, Union

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


@external_store(
    group="datasets",
    root_dir=os.getenv("QUILT_ROOT_DIR", MISSING),
    subsets=["openpath", "quilt"],
)
class Quilt(Dataset[Example]):
    """Quilt-1M dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing json files with data entries.
    split : {"train", "val"}
        Dataset split.
    subsets : List[str], optional, default=["openpath", "pubmed", "quilt", "laion"]
        Subsets of Quilt-1M to load.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "val"] = "train",
        subsets: Optional[List[str]] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        # input validation
        if not os.path.exists(root_dir):
            raise RuntimeError(f"Root directory is not accessible: {root_dir}.")
        if subsets is None:
            subsets = ["openpath", "pubmed", "quilt", "laion"]

        # read entries
        self.data_df = pd.read_csv(os.path.join(root_dir, f"quilt_1m_{split}.csv"))
        # drop unnecessary and space-consuming columns
        self.data_df.drop(
            columns=[
                "noisy_text",
                "corrected_text",
                "med_umls_ids",
                "roi_text",
                "Unnamed: 0",
            ],
            inplace=True,
        )
        # filter entries based on `subset`
        self.data_df = self.data_df.loc[
            self.data_df.apply(lambda row: row["subset"] in subsets, axis=1)
        ]

        # the 'pathology' column is a list of strings
        self.data_df["pathology"] = self.data_df["pathology"].apply(self._safe_eval)

        self.root_dir = root_dir
        self.subsets = subsets

        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.tokenizer = tokenizer

    def _safe_eval(self, x: str) -> list[str]:
        """Safely evaluate a string as a list of strings."""
        if pd.isna(x):
            return []
        try:
            return ast.literal_eval(x)  # type: ignore[no-any-return]
        except (ValueError, SyntaxError):
            return []

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        try:
            img_path = os.path.join(
                self.root_dir, "quilt_1m", self.data_df["image_path"].iloc[idx]
            )
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image for entry {idx}: image_path={img_path}", e)
            idx = (idx + 1) % len(self.data_df.index)
            return self.__getitem__(idx)
        caption = self.data_df["caption"].iloc[idx]

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
                "qid": self.data_df.index[idx],
                "magnification": self.data_df["magnification"].iloc[idx],
                "height": self.data_df["height"].iloc[idx],
                "width": self.data_df["width"].iloc[idx],
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT.name in tokens
                ), f"Missing key `{Modalities.TEXT.name}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT.name] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data_df.index)
