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
    include_extra: bool, default=False
        Whether or not to include the additional data samples extracted by us
        in October 2024.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        include_extra: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        data_path = os.path.join(root_dir, f"{split}.jsonl")
        with open(data_path, encoding="utf-8") as file:
            entries = [json.loads(line) for line in file.readlines()]

        # convert relative image paths to absolute paths
        for entry in entries:
            entry["subfig_path"] = os.path.join(root_dir, "images", entry["image"])

        if include_extra:
            data_path = os.path.join(root_dir, f"pmc_oa2_{split}.jsonl")
            with open(data_path, encoding="utf-8") as file:
                entries.extend([json.loads(line) for line in file.readlines()])

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
            with Image.open(entry["subfig_path"]) as img:
                image = img.convert("RGB")
        except Exception as e:
            print(
                f"Error loading image for entry {idx}: image_path={entry['subfig_path']}",
                e,
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)
        caption = entry["caption"]

        if self.transform is not None:
            image = self.transform(image)

        tokens = self.tokenizer(caption) if self.tokenizer is not None else None

        example = Example(
            {
                Modalities.RGB.name: image,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
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
        return len(self.entries)
