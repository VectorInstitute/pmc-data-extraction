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


@external_store(group="datasets", root_dir="/datasets/PMC-15M/filtered_biomedica/filtered_v4")
class BiomedicaFiltered(Dataset[Example]):
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
    use_full_caption : bool, default=False
        Use full captions or not.
    transform : Optional[Callable], default=None
        Transform applied to images.
    tokenizer : Optional[Callable], default=None
        Function applied to textual captions.
    """

    def __init__(
        self,
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        mode = "Whole",
        include_extra: bool = False,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        tokenizer: Optional[
            Callable[[str], Union[torch.Tensor, Dict[str, torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initialize the dataset."""
        # data_path = os.path.join(f"/projects/DeepLesion/datasets/data_reps/{split}_with_clip_score.jsonl")
        # data_path = os.path.join(root_dir, "filtered_only_image_caption.jsonl")
        # print(f"Loading {data_path}...")
        # with open(data_path, encoding="utf-8") as file:
            # entries = [json.loads(line) for line in file.readlines()]
        
        data_path = os.path.join(root_dir, f"image_files_list.txt")
        print(f"Loading {data_path}...")
        with open(data_path, "r") as f:
            entries = [line.strip() for line in f]

        print(f"AFTER READING {data_path}...")

        # convert relative image paths to absolute paths
        # for entry in entries:
            # entry["subfig_path"] = os.path.join("/projects/multimodal/datasets/final_pmc_oa/enriched/images", entry["image"])

        # if include_extra:
        #     data_path = os.path.join(root_dir, f"pmc_oa2_{split}.jsonl")
        #     with open(data_path, encoding="utf-8") as file:
        #         entries.extend([json.loads(line) for line in file.readlines()])
                
        # if mode == "CLIP":
        #     entries = [entry for entry in entries if float(entry.get("CLIPScore", 0)) > 0.46]
        print(f"--------------------------------------- {len(entries)} ------------------------------------------------------------")

        self.entries = entries

        self.root_dir = root_dir

        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.tokenizer = tokenizer
        # self.use_full_caption = use_full_caption

    def __getitem__(self, idx: int) -> Example:
        """Return the idx'th data sample."""
        # entry = self.entries[idx]
        # subfig_path = os.path.join(self.root_dir, entry["subfig_path"])
        # subfig_path = entry["image_path"]
        subfig_path = self.entries[idx]
        caption_path = os.path.splitext(subfig_path)[0] + ".txt"
        try:
            with Image.open(subfig_path) as img:
                image = img.convert("RGB")
            with open(caption_path, "r") as f:
                caption = f.read().strip()
        except Exception as e:
            print(
                f"Error loading image for entry {idx}: image_path={subfig_path}",
                e,
            )
            idx = (idx + 1) % len(self.entries)
            return self.__getitem__(idx)
        # if self.use_full_caption:
        #     caption = entry["full_caption"]
        # else:
        #     caption = entry["sub_caption"]
        # caption = entry["caption"]
        if len(caption) == 0:
            print(
                f"Empty caption for entry {idx}: image_path={subfig_path}, caption={caption}"
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
                Modalities.TEXT.target: 0,
                "image_path": subfig_path,
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