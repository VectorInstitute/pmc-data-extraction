"""PMC-VL Dataset."""

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


Image.MAX_IMAGE_PIXELS = None


@external_store(group="datasets", root_dir=os.getenv("PMCVL_ROOT_DIR", MISSING))
class PMCVL(Dataset[Example]):
    """PMC-VL dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "test"}
        Dataset split.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "test"] = "train",
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
            img_path = os.path.join(self.root_dir, "figures", entry["media_name"])
            cap_path = os.path.join(self.root_dir, "captions", entry["caption_name"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")
            with open(cap_path, encoding="utf-8") as file:
                caption = file.read()
        except Exception:
            print(
                f"Error loading image or caption for entry {idx}: image_path={img_path} caption_path={cap_path}"
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)

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
