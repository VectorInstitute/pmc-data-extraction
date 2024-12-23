"""Aggregate OpenPMC-VL entries of all volumes, the split them into train-val-test."""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def load_jsonl(filename: str) -> List[Dict[str, str]]:
    """Load a dictionary from jsonl file.

    Parameters
    ----------
    filename: str
        Path to the jsonl file containing the dictionary.

    Returns
    -------
    entries: List[Dict[str, str]]
        Loaded dictionary from the file.
    """
    with open(filename, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file.readlines()]  # noqa: RET504
    return entries  # noqa: RET504


def save_jsonl(data: List[Dict[str, str]], filename: str) -> None:
    """Save given data in jsonl format.

    Parameters
    ----------
    data: List[Dict[str, str]]
        Dictionary to be stored in file.
    filename: str
        File name.
    """
    with open(filename, "w") as outfile:
        for sample in data:
            json.dump(sample, outfile)
            outfile.write("\n")


def aggregate_volumes(
    jsonl_rootdir: str, accepted_exts: List[str]
) -> List[Dict[str, str]]:
    """Gather all volumes in one file, don't include non-image files.

    Parameters
    ----------
    jsonl_rootdir: str
        Path to the directory containing entry volumes as jsonl files.
    accepted_exts: List[str]
        List of accepted extensions (for example, jpg, png, etc.) to include in the
        aggregated list of entries.

    Returns
    -------
    all_data: List[Dict[str, str]]
        List of all entries from all volumes.
    """
    # list all available volumes
    volumes = [
        file
        for file in os.listdir(jsonl_rootdir)
        if re.fullmatch(r"[0-9]+.*\.jsonl", file) is not None
    ]
    volumes = sorted(
        volumes, key=lambda x: int(x.replace(".jsonl", "").replace("_clean", ""))
    )
    print(f"Found volumes: {volumes}")

    all_data = []  # list for aggregated dataset
    for volume in volumes:
        # load volume
        entries = load_jsonl(os.path.join(jsonl_rootdir, volume))

        n_nonimage = 0  # count number of non-image media files
        for entry in tqdm(
            entries, total=len(entries), desc=f"processing volume {volume}"
        ):
            # check if media is image
            extension = entry["media_name"].split(".")[-1]
            if extension not in accepted_exts:
                n_nonimage += 1
                continue
            all_data.append(entry)

        print(f"{n_nonimage} non-image samples were in volume {volume}")
        print(f"{len(all_data)} samples aggregated from volume {volume}.")

    return all_data


def train_test_split(
    agg_filename: str,
    train_ratio: float,
    test_ratio: Optional[float] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Split aggregated entry list into train and test sets.

    Parameter
    ---------
    agg_filename: str
        Path to the jsonl file containing aggregated entries of all volumes.
    train_ratio: float
        Ratio of the train split.
    test_ratio: float, optional, default=None
        Ratio of the test split. By default, it's the complement of the train ratio.
    seed: int, default=42
        Random seed used to permute entries before splitting.

    Return
    ------
    train_data: List[Dict[str, str]]
        List of entries assigned to the train split.
    test_data: List[Dict[str, str]]
        List of entries assigned to the test split.
    """
    # load dataset
    print(f"Loading aggregated data: {agg_filename} ...")
    entries = load_jsonl(agg_filename)

    # determine number of train and test samples
    print("Determining number of train and test samples...")
    ntrain = int(len(entries) * train_ratio)
    if test_ratio is not None:
        ntest = int(len(entries) * test_ratio)
    else:
        ntest = len(entries) - ntrain

    # randomly permute data
    print("Permuting data samples...")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(entries)  # type: ignore[arg-type]

    # split dataset
    train_data = perm[:ntrain]
    test_data = perm[ntrain : (ntrain + ntest)]
    print("Finished splitting:")
    print(f"num train: {len(train_data)}")
    print(f"num test: {len(test_data)}")

    return train_data, test_data  # type: ignore[return-value]


def create_dummy(filename: str, nsamples: int = 1000) -> None:
    """Create small sample sets for model debugging purposes.

    Parameters
    ----------
    filename: str
        Name of the jsonl file (i.e. entry list) to sample.
    nsamples: int
        Number of samples.
    """
    # load dataset
    print(f"Loading original data: {filename}...")
    data = load_jsonl(filename)
    # save dummy version
    save_jsonl(data[:nsamples], filename.replace(".jsonl", "_dummy.jsonl"))


def unsplit(filename1: str, filename2: str, outputfilename: str) -> None:
    """Aggregate splits into a single file again.

    This function can be used to redo splitting.

    Parameters
    ----------
    filename1: str
        Path to the first split's jsonl file.
    filename2: str
        Path to the second split's jsonl file.
    outputfilename: str
        Path to the jsonl file where the aggregated entries are stored.
    """
    print(f"Loading data: {filename1}...")
    data1 = load_jsonl(filename1)

    print(f"Loading data: {filename2}...")
    data2 = load_jsonl(filename2)

    print("Aggregating two files...")
    data1.extend(data2)

    print(f"Saving to new file: {outputfilename}")
    save_jsonl(data1, outputfilename)
    print(f"Aggregated data saved to {outputfilename}")


def main(jsonl_rootdir: str, accepted_exts: List[str]) -> None:
    """Aggregate and split entries.

    New train-val-test splits will be stored in `jsonl_rootdir` as jsonl files.

    Parameters
    ----------
    jsonl_rootdir: str
        Path to the directory containing entry volumes as jsonl files.
    accepted_exts: List[str]
        List of accepted extensions (for example, jpg, png, etc.) to include
        in the aggregated list of entries.
    """
    # aggregate
    agg_filename = os.path.join(jsonl_rootdir, "aggregated.jsonl")
    agg_data = aggregate_volumes(jsonl_rootdir, accepted_exts)
    save_jsonl(agg_data, agg_filename)

    # split
    train_data, test_data = train_test_split(agg_filename, train_ratio=0.8, seed=42)
    save_jsonl(train_data, os.path.join(jsonl_rootdir, "train_clean.jsonl"))
    save_jsonl(test_data, os.path.join(jsonl_rootdir, "test_clean.jsonl"))

    test_filename = os.path.join(jsonl_rootdir, "test_clean.jsonl")
    val_data, test_data = train_test_split(test_filename, train_ratio=0.75, seed=42)
    save_jsonl(val_data, os.path.join(jsonl_rootdir, "val_clean.jsonl"))
    save_jsonl(test_data, os.path.join(jsonl_rootdir, "test_clean.jsonl"))

    # remove intermediate file
    os.remove(agg_filename)


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl-rootdir",
        help="path to the directory containing entry volumes as jsonl files.",
        default=".",
    )
    parser.add_argument(
        "--accepted-exts",
        help="list of accepted extensions to include in the splits.",
        nargs="+",
        default=["jpg", "png"],
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.jsonl_rootdir, args.accepted_exts)
