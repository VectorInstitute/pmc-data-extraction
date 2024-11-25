"""Download and extract image-caption pairs from PMC Open Access Subset.

Commandline code to run this function:
```bash
python src/fetch_oa.py --extraction-dir path/to/output/directory
```
"""

import logging
import os
import pathlib
import shutil
import subprocess
import sys
from argparse import Namespace
from parser.parse_oa import get_volume_info
from typing import Dict, List, Tuple

from args import parse_args_oa
from data import OA_LINKS
from tqdm import tqdm
from utils import read_jsonl, write_jsonl


def create_logger() -> Tuple[logging.Logger, logging.StreamHandler]:  # type: ignore[type-arg]
    """Set up logger and console handler."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - \33[32m%(message)s\033[0m"
    )
    console_handler.setFormatter(formatter)
    return logger, console_handler


def extract_archive(archive_path: str, target_dir: str) -> None:
    """Extract article archive.

    Parameters
    ----------
    archive_path: str
        Path to the archive of xml articles.
    target_dir: str
        Target directory to store extracted contents.
    """
    subprocess.call(["tar", "zxf", archive_path, "-C", target_dir])


def download_archive(
    args: Namespace, logger: logging.Logger, volumes: List[int]
) -> None:
    """Download xml archives of requested volumes.

    Parameters
    ----------
    args: argparse.Namespace
        Commandline arguments.
    logger: logging.Logger
        Logger to log information and errors to console.
    volumes: List[int]
        List of volumes to download.
    """
    logger.info("Volumes to download: %s" % volumes)

    for volume_id in volumes:
        volume = "PMC0%02dxxxxxx" % volume_id
        csv_url = OA_LINKS[args.license_type][volume]["csv_url"]
        tar_url = OA_LINKS[args.license_type][volume]["tar_url"]
        logger.info(csv_url)
        logger.info(tar_url)

        subprocess.call(
            [
                "wget",
                "-nc",
                "-nd",
                "-c",
                "-q",
                "-P",
                "%s/%s" % (args.extraction_dir, volume),
                csv_url,
            ]
        )
        subprocess.call(
            [
                "wget",
                "-nc",
                "-nd",
                "-c",
                "-q",
                "-P",
                "%s/%s" % (args.extraction_dir, volume),
                tar_url,
            ]
        )

        if not pathlib.Path(
            "%s/%s/%s" % (args.extraction_dir, volume, volume)
        ).exists():
            logger.info("Extracting %s..." % volume)
            extract_archive(
                archive_path="%s/%s/%s"
                % (args.extraction_dir, volume, tar_url.split("/")[-1]),
                target_dir="%s/%s" % (args.extraction_dir, volume),
            )
            logger.info("%s done.", volume)
        else:
            logger.info("%s already exists.", volume)


def download_media(args: Namespace, volume_info: List[Dict[str, str]]) -> None:
    """Download media.

    Media may be image, video, pdf, docx, etc.
    In the current implementation of the module, only images are downloaded.

    Parameters
    ----------
    args: argparse.Namespace
        Commandline arguments.
    volume_info: List[Dict[str, str]]
        List of <img, caption> pairs extracted from the volumes.
        For each <img, caption> pair, a dictionary is created containing the
        following keys:
            - PMC_ID: This is a unique ID assigned to each article by PubMed.
            - media_id: Tag ID of the image or other media in the article's
                        xml file.
            - caption: Caption of the media.
            - media_url: URL from which media is downloaded using `wget`.
            - media_name: Filename of the downloaded media.
    """
    # make directory to store figures (and other media)
    figures_dir = f"{args.extraction_dir}/figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir, 0o755)

    # download figures (and other media)
    for obj in tqdm(volume_info, desc="download media"):
        media_url = obj["media_url"]
        media_name = obj["media_name"]
        file_path = f"{figures_dir}/{media_name}"

        # BUG connection issues could result in nothing downloaded
        # BUG wget alone results in 403 forbidden
        subprocess.call(
            [
                "wget",
                "-U",
                "Mozilla/5.0 (X11; Linux x86_64; rv:78.0) Gecko/20100101 Firefox/78.0",
                "-nc",
                "-nd",
                "-c",
                "-q",
                "-P",
                file_path,
                media_url,
            ]
        )
        if not os.path.exists(file_path):
            print(
                "ERROR: download failed, use the following command to check "
                f"connection, file_path = {file_path}, media_url = {media_url}"
            )


def main() -> None:
    """Entry point for the openpmcvl foundation module."""
    # set up logger and console handler
    logger, _ = create_logger()

    # check if wget is available
    if not shutil.which("wget"):
        print("`wget` not found, please install `wget` and put it on your PATH.")
        sys.exit(-1)

    # get commandline args
    args = parse_args_oa()
    print(args)

    # download xml articles
    download_archive(args=args, logger=logger, volumes=args.volumes)

    # set filename where volume info is stored
    save_name = "".join([str(volume_id) for volume_id in args.volumes])
    volume_info_path = f"{args.extraction_dir}/{save_name}.jsonl"

    if not os.path.exists(volume_info_path):
        # parse xml files for <img, caption> pairs
        logger.info("Extracting Volume Info")
        volume_info = get_volume_info(
            args=args, volumes=args.volumes, extraction_dir=args.extraction_dir
        )

        # save <img, caption> pairs of the volume in jsonl file
        logger.info("Saving Volume Info...")
        write_jsonl(data_list=volume_info, save_path=volume_info_path)
        logger.info("Saved")
    else:  # volume_info already extracted
        volume_info = read_jsonl(file_path=volume_info_path)

    # download media mentioned in volume_info
    logger.info("Downloading media files...")
    download_media(args, volume_info)
    logger.info("Done")


if __name__ == "__main__":
    main()
