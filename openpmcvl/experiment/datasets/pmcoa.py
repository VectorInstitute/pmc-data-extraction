"""PMC-OA Dataset."""

import json
import os
from typing import Callable, Dict, Literal, Optional, Union

import torch
from mmlearn.conf import external_store
from mmlearn.constants import EXAMPLE_INDEX_KEY
from mmlearn.datasets.core import Modalities
from mmlearn.datasets.core.example import Example
from omegaconf import MISSING
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


@external_store(group="datasets", root_dir=os.getenv("PMCOA_ROOT_DIR", MISSING))
class PMCOA(Dataset[Example]):
    """PMC-OA dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        data_path = os.path.join(root_dir, f"{split}.jsonl")
        with open(data_path, encoding="utf-8") as file:
            entries = [json.loads(line) for line in file.readlines()]
        self.entries = entries

        self.root_dir = root_dir

        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.tokenizer = tokenizer

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        entry = self.entries[idx]
        try:
            img_path = os.path.join(self.root_dir, "images", entry["image"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        except Exception:
            print(
                f"Error loading image for entry {idx}: image_path={img_path}"
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)
        caption = entry["caption"]

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        example = Example(
            {
                Modalities.RGB: image,
                Modalities.TEXT: caption,
                EXAMPLE_INDEX_KEY: idx,
            }
        )

        if tokens is not None:
            if isinstance(tokens, dict):  # output of HFTokenizer
                assert (
                    Modalities.TEXT in tokens
                ), f"Missing key `{Modalities.TEXT}` in tokens."
                example.update(tokens)
            else:
                example[Modalities.TEXT] = tokens

        return example

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.entries)
