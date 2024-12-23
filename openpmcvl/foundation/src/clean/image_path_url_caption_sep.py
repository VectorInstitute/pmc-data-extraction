"""Fix image paths in the jsonl files and remove non-existent images.

This is the first step of data cleaning after downloading image-caption pairs
from Pubmed articles. For each downloaded volume of image-caption pairs, we
have a jsonl file with metadata of image path, image url, caption, etc.
If an image url was not found in an article, a placeholder url ("https://null.jpg")
is stated in the jsonl file. Moreover, some image paths might actually point to
a directory containing the actual image.
This script fixes the image paths to point to the actual image file and removes
all entries whose image url is the placeholder.
"""

import argparse
import json
import os
from typing import Dict, List

from tqdm import tqdm


FAKE_URL = "https://null.jpg"


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


def remove_placeholder_url(jsonl_file: str) -> List[Dict[str, str]]:
    """Remove entries whose image url is a placeholder.

    Currently, this package uses "https://null.jpg" as placeholder url.

    Parameters
    ----------
    jsonl_file: str
        Path to the jsonl file of the intended volume to clean.

    Return
    ------
    clean_entries: List[Dict[str, str]]
        Cleaned entries sans the placeholder ones.
    """
    # load entries
    entries = load_jsonl(jsonl_file)

    # remove placeholder images
    clean_entries = []
    for entry in tqdm(entries, total=len(entries), desc=f"cleaning {jsonl_file}"):
        if entry["media_url"] == FAKE_URL:
            continue
        clean_entries.append(entry)
    print(
        f"{len(entries) - len(clean_entries)}/{len(entries)} entries with placeholder url were removed."
    )

    return clean_entries


def fix_media_name(
    root_dir: str, entries: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Fix `media_name` in entries.

    The `media_name` in loaded entries of OpenPMC-VL might point to a
    directory instead of a media file. Rewrite this field to point to
    the media file inside the directory instead.

    Parameters
    ----------
    root_dir: str
        Path to where the jsonl files and the "figures" folder are located.
    entries: List[Dict[str, str]]
        Entries of given volume with `media_name`.

    Returns
    -------
    entries: List[Dict[str, str]]
        Cleaned entries with fixed `media_name`.
    """
    for entry in tqdm(entries, total=len(entries), desc="fixing media_name"):
        og_medianame = os.path.join(root_dir, "figures", entry["media_name"])
        if os.path.isfile(og_medianame):
            continue
        inname = entry["media_url"].split("/")[-1]
        entry["media_name"] = os.path.join(entry["media_name"], inname)
        if not os.path.isfile(os.path.join(root_dir, "figures", entry["media_name"])):
            print(
                "Error: media_name still doesn't point to a file.", entry["media_name"]
            )

    return entries


def separate_captions(
    cap_rootdir: str, entries: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Separate captions from jsonl files into text files.

    Store captions in a single folder as text files, and put the path
    to the text files in jsonl files instead of the actual caption.

    Parameters
    ----------
    cap_rootdir: str
        Path to where the caption text files will be stored.
    entries: List[Dict[str, str]]
        Data entries of a given volume.

    Returns
    -------
    entries: List[Dict[str, str]]
        Cleaned entries with paths to caption text files instead of actual captions.
    """
    for entry in tqdm(entries, total=len(entries), desc="separating captions"):
        # write caption in a separate file
        cap_filename = f"{entry['PMC_ID']}_{entry['media_id']}.txt"
        with open(os.path.join(cap_rootdir, cap_filename), "w") as f:
            f.write(entry["caption"])

        # replace caption with caption file name in jsonl
        entry.pop("caption", None)
        entry["caption_name"] = cap_filename

    return entries


def main(license_dir: str, volumes: List[int], sep_captions: bool = True) -> None:
    """Clean given volumes of a given license.

    Parameters
    ----------
    license_dir: str
        Directory where original jsonl files of the downloaded papers under a
        certain license are stored.
    volumes: List[int]
        Index of the volumes to be cleaned. A file named "{volume}.jsonl" is
        expected to exist in `license_dir`.
    sep_captions: bool, default = True
        Whether or not to store captions in separate text files instead of in
        jsonl files.
    """
    # create directory for cleaned jsonl files
    outdir = os.path.join(license_dir, "processed")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # create captions dir if required
    if sep_captions:
        cap_rootdir = os.path.join(license_dir, "captions")
        if not os.path.isdir(cap_rootdir):
            os.mkdir(cap_rootdir)

    for volume in volumes:
        # clean data
        jsonl_file = os.path.join(license_dir, f"{volume}.jsonl")
        clean_entries = remove_placeholder_url(jsonl_file)
        clean_entries = fix_media_name(license_dir, clean_entries)
        if sep_captions:
            clean_entries = separate_captions(cap_rootdir, clean_entries)

        # save cleaned data
        outfile = os.path.join(outdir, f"{volume}_clean.jsonl")
        save_jsonl(clean_entries, outfile)


def parse_arguments() -> argparse.Namespace:
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--license-dir",
        help="directory where original jsonl files of the downloaded papers (under a certain license) are stored.",
        default=".",
    )
    parser.add_argument(
        "--volumes",
        help="determine the volumes to clean",
        nargs="+",
        default=[1],
        type=int,
    )
    parser.add_argument(
        "--no-sep-captions",
        dest="sep_captions",
        help="whether or not to store captions in separate text files instead of in jsonl files.",
        action="store_false",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    main(args.license_dir, args.volumes, sep_captions=args.sep_captions)
