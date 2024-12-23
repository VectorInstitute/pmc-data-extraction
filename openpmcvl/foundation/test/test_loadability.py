"""Try loading all files listed in OpenPMC-VL and remove erroneous ones."""

import argparse
import json
import os
import sys
from typing import Dict, List

import multiprocess as mp
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None


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


def remove_faulty_files(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Try loading all files in the given split and remove erroneous ones.

    Parameters
    ----------
    entries: List[Dict[str, str]]
        List of entries in a given split to check.

    Return
    ------
    clean_entries: List[Dict[str, str]]
        List of entries in the given split that loaded successfully.
    """
    clean_entries = []
    # load image and captions
    for entry in tqdm(
        entries, total=len(entries), desc=f"cleaning {input_split} split"
    ):
        try:
            img_path = os.path.join(root_dir, "figures", entry["media_name"])
            cap_path = os.path.join(root_dir, "captions", entry["caption_name"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")  # noqa: F841
            with open(cap_path, encoding="utf-8") as file:
                caption = file.read()  # noqa: F841
            clean_entries.append(entry)
        except Exception as e:
            print(
                f"Error loading image or caption: image_path={img_path} caption_path={cap_path}",
                "\n",
                e,
                "\nRemoving entry from entrylist...",
            )
    print(f"{len(entries) - len(clean_entries)} entries were removed.")
    return clean_entries


def main(input_split: str, clean_split: str) -> None:
    """Try loading all files in the given split and remove erroneous ones.

    Parameters
    ----------
    input_split: str
        Name of the input split.
    clean_split: str
        Name of the resulting split where only loadable files exist.
    """
    # load split
    entries = load_jsonl(os.path.join(root_dir, f"{input_split}.jsonl"))

    # remove faulty files
    clean_entries = remove_faulty_files(entries)

    # save clean entrylist
    print("Saving clean entrylist...")
    filename = os.path.join(root_dir, f"{clean_split}.jsonl")
    save_jsonl(clean_entries, filename)
    print(f"Saved clean entrylist in {filename}")


def main_parallel(input_split: str, clean_split: str, nprocess: int) -> None:
    """Try loading all files in the given split and remove erroneous ones.

    This function runs on multiple CPU cores in parallel.

    Parameters
    ----------
    input_split: str
        Name of the input split.
    clean_split: str
        Name of the resulting split where only loadable files exist.
    nprocess: int
        Number of parallel processes. Ideally, this is the number of CPU cores
        on the compute node.
    """
    # load split
    print(f"Loading {input_split} split...")
    entries = load_jsonl(os.path.join(root_dir, f"{input_split}.jsonl"))

    # slice entries to the number of tasks
    print("Distributing entries...")
    sublength = (len(entries) + nprocess) // nprocess
    args = []
    for idx in range(0, len(entries), sublength):
        args.append(entries[idx : (idx + sublength)])

    # run jobs in parallel
    with mp.Pool(processes=nprocess) as pool:
        results = pool.map(
            remove_faulty_files, args
        )  # list x list x dictionary == nprocess x entries per process x entry

    # aggregate results
    clean_entries = []
    for proc in results:
        clean_entries.extend(proc)

    # save clean entrylist
    print("Saving clean entrylist...")
    filename = os.path.join(root_dir, f"{clean_split}.jsonl")
    save_jsonl(clean_entries, filename)
    print(f"Saved clean entrylist in {filename}")


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-dir",
        help="root directory of the dataset; i.e. where jsonl files of the splits are stored.",
        default=".",
    )
    parser.add_argument(
        "--input-split",
        help="split to test.",
        default="test_clean",
    )
    parser.add_argument(
        "--clean-split",
        help="name of the resulting split containing only successfully loaded entries.",
        default="test_cleaner",
    )
    parser.add_argument(
        "--mode",
        help="whether to run the script on a single core or parallel cores.",
        default="parallel",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cmd_args = parse_arguments()
    root_dir = cmd_args.root_dir
    input_split = cmd_args.input_split
    clean_split = cmd_args.clean_split
    assert (
        root_dir is not None
    ), "Please enter root directory of OpenPMC-VL dataset in `PMCVL_ROOT_DIR` environment variable."

    if cmd_args.mode == "single":
        # single core
        main(input_split, clean_split)
    elif cmd_args.mode == "parallel":
        # multi core
        nprocess = os.environ.get("SLURM_CPUS_PER_TASK")
        if nprocess is None:
            print(
                "Please set the number of CPUs in environment variable `SLURM_CPUS_PER_TASK`."
            )
            sys.exit()
        main_parallel(input_split, clean_split, nprocess=int(nprocess))
    else:
        print("mode is not accepted; enter either 'single' or 'parallel'.")
