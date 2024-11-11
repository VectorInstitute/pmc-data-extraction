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


@external_store(group="datasets", root_dir=os.getenv("PMCOA_3_ROOT_DIR", MISSING))
class PMCOA_4(Dataset[Example]):
    """PMC-OA dataset.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing the CSV file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    modality : Optional[int], default=None
        If specified, only entries with this label will be loaded.
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
        include_extra: bool = True,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        data_path = os.path.join(root_dir, f"pmcoa_2_{split}_imagenet_5_labels.jsonl")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"{data_path} does not exist.")

        with open(data_path, encoding="utf-8") as file:
            entries = [json.loads(line) for line in file.readlines()]

        # Update paths to be absolute
        for entry in entries:
            entry["subfig_path"] = os.path.join(root_dir, "images", entry["image_path"])

        self.entries = entries
        self.root_dir = root_dir
        self.transform = ToTensor() if transform is None else transform
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
                # Modalities.RGB.name: image,
                Modalities.TEXT.name: caption,
                EXAMPLE_INDEX_KEY: idx,
                "label": entry["modality_label"],
                entry["modality"].lower(): image,
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
