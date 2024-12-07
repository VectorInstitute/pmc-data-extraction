"""Clean PMC-2M with summarized inline refs dataset."""
import json
import os
from typing import Literal
from tqdm import tqdm


def clean_pmc2m_sum(
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train") -> None:
    """Clean PMC-2M with summarized inline refs dataset.

    Cleaning entails below steps:
    1. Create `image_fullpath` key.
    2. Remove entries with no caption or intext refs summary.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    """
    # load entries
    data_path = os.path.join(root_dir, "clean", f"{split}.jsonl")
    with open(data_path, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file.readlines()]

    # convert relative image paths to absolute paths
    for entry in tqdm(entries):
        entry["image_fullpath"] = os.path.join(root_dir, "images", entry["image"])

    # check text existence
    clean_entries = []
    for entry in tqdm(entries):
        if entry["caption"] is None:
            print(f"caption not string: {entry['caption']}")
            entry["caption"] = ""
        if entry["intext_refs_summary"] is None:
            print(f"intext_refs_summary not string: {entry['intext_refs_summary']}")
            entry["intext_refs_summary"] = ""
        if entry["caption"] == "" and entry["intext_refs_summary"] == "":
            continue
        clean_entries.append(entry)
    print(f"{len(entries) - len(clean_entries)} entries removed due to non-existent caption and intext reference.")

    # write clean entries
    filename = os.path.join(root_dir, "clean", f"{split}_clean.jsonl")
    with open(filename, "w") as outfile:
        for entry in clean_entries:
            json.dump(entry, outfile)
            outfile.write("\n")
    print(f"Saved {len(clean_entries)} entries in {filename}")


def separate_captions(
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train"):
    """Store captions in separate files.

    Load captions in each call to __getitem__ to reduce GPU memory usage.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    """
    # load entries
    data_path = os.path.join(root_dir, "clean", f"{split}.jsonl")
    with open(data_path, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file.readlines()]

    # separate caption
    sep_entries = []
    for entry in tqdm(entries):
        caption = " ".join([entry["caption"], entry["intext_refs_summary"]])
        caption_filename = os.path.join(root_dir, "captions2", entry["image"].replace("jpg", "txt"))
        with open(caption_filename, "w") as outfile:
            outfile.write(caption)
        sep_entries.append({"image_fullpath": entry["image_fullpath"], "caption_fullpath": caption_filename})

    # write sep entries
    filename = os.path.join(root_dir, "clean", f"{split}_sep.jsonl")
    with open(filename, "w") as outfile:
        for entry in tqdm(sep_entries):
            json.dump(entry, outfile)
            outfile.write("\n")
    print(f"Saved {len(sep_entries)} entries in {filename}")


def clean_pmcoa_2(
        root_dir: str,
        split: Literal["train", "valid", "test"] = "train",
        include_extra: bool = False) -> None:
    """Clean PMC-OA-2 dataset.

    Cleaning entails below step:
    1. Remove entries with no caption or intext refs summary.

    Parameters
    ----------
    root_dir : str
        Path to the root folder containing jsonl file with data entries.
    split : {"train", "valid", "test"}
        Dataset split.
    include_extra: bool, default=False
        Whether or not to include the additional data samples extracted by us
        in October 2024.
        **Note:** If you set this to True, the resulting jsonl file will contain the
        cleaned entries of both PMC-OA and PMC-OA-2 additional images.
    """
    # load entries
    data_path = os.path.join(root_dir, f"{split}.jsonl")
    with open(data_path, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file.readlines()]

    if include_extra:
        data_path = os.path.join(root_dir, f"pmc_oa2_{split}.jsonl")
        with open(data_path, encoding="utf-8") as file:
            entries.extend([json.loads(line) for line in file.readlines()])

    # check text existence
    clean_entries = []
    for entry in tqdm(entries):
        if entry["caption"] is None or entry["caption"] == "":
            continue
        clean_entries.append(entry)
    print(f"{len(entries) - len(clean_entries)} entries removed due to non-existent caption.")

    # write clean entries
    filename = os.path.join(root_dir, f"{split}_clean.jsonl")
    with open(filename, "w") as outfile:
        for entry in clean_entries:
            json.dump(entry, outfile)
            outfile.write("\n")
    print(f"Saved {len(clean_entries)} entries in {filename}")




if __name__ == "__main__":
    # root_dir = os.getenv("PMC2M_SUMM_ROOT_DIR", "")
    # for split in ["train"]:
    #     print(f"PROCESSING SPLIT {split}...")
    #     clean_pmc2m_sum(root_dir, split)
    #     separate_captions(root_dir, f"{split}_clean")

    root_dir = os.getenv("PMCOA_ROOT_DIR", "")
    split = "train"
    clean_pmcoa_2(root_dir, split, include_extra=False)
