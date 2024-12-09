"""Parse PMC Open Access articles.

Parsing includes below steps:
1. Request PMC pages.
2. Extract <image, caption> pairs from pages.
"""

import codecs
import os
import pathlib
import shutil
import subprocess
import time
from argparse import Namespace
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup
from data import UPDATE_SCHEDULE
from tqdm import tqdm


def get_img_url(pmc_id: str, fig_id: str, max_retries: int = 10) -> str:
    """Get download URL of a given image in a given article.

    This URL could be used to download the image with `wget`.
    To find the final URL, an intermediary html page is downloaded and parsed;
    to make sure this page gets downloaded, we implement a retry strategy
    where the number of retries is given by an argument and the wait time
    between retries grows exponentially.

    Parameters
    ----------
    pmc_id: str
        PMC_ID of the article.
    fig_id: str
        Tag ID of the desired image in the article's xml file.
    max_retries: int, default = 10
        Maximum number of retires to download an intermediary page
        to find the image URL.

    Returns
    -------
    img_url: str
        Download URL of the image.
    """
    img_src_url = "https://pmc.ncbi.nlm.nih.gov/articles/%s/figure/%s/" % (
        pmc_id,
        fig_id,
    )
    file_path = f"/datasets/PMC-15M/temp/{pmc_id}_{fig_id}"
    for i in range(max_retries):
        returncode = subprocess.call(
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
                img_src_url,
            ]
        )
        if returncode == 0:
            break
        time.sleep(2**i)
    # find the actual image url in the xml
    try:
        xml_path = os.path.join(file_path, "index.html")
        with codecs.open(xml_path, encoding="utf-8") as f:
            document = f.read()
        soup = BeautifulSoup(document, "lxml")
        img = soup.find(name="img", attrs={"class": "graphic"})
        img_url = str(img.attrs["src"])
        # remove temporary downloaded page
        shutil.rmtree(file_path)
    except Exception as e:
        print(f"ERROR: Problem in extracting image {file_path}", e)
        img_url = "https://null.jpg"
    return img_url


def parse_xml(args: Namespace, xml_path: str) -> List[Dict[str, str]]:
    """Extract <img, caption> pairs of one article.

    Parameters
    ----------
    args: argparse.Namespace
        Commandline arguments for the whole module.
    xml_path: str
        Path to xml file containing the full article text.

    Returns
    -------
    item_info: List[Dict[str, str]]
        List of <img, caption> pairs extracted from the article.
        For each <img, caption> pair, a dictionary is created containing the
        following keys:
            - PMC_ID: This is a unique ID assigned to each article by PubMed.
            - media_id: Tag ID of the image or other media in the article's
                        xml file.
            - caption: Caption of the media.
            - media_url: URL from which media is downloaded using `wget`.
            - media_name: Filename of the downloaded media.
    """
    with codecs.open(xml_path, encoding="utf-8") as f:
        document = f.read()
    soup = BeautifulSoup(document, "lxml")

    # extract PMC_ID from xml_path
    if isinstance(xml_path, pathlib.Path):
        xml_path = str(xml_path)
    pmc_id = xml_path.split("/")[-1].strip(".xml")

    item_info = []

    figs = soup.find_all(name="fig")
    for fig in figs:
        if "id" in fig.attrs:
            media_id = fig.attrs["id"]
        else:
            continue

        if fig.graphic:
            _ = fig.graphic.attrs["xlink:href"]
            media_url = get_img_url(pmc_id, media_id, args.num_retries)
            file_extension = media_url.split(".")[-1]  # .jpg
            media_name = f"{pmc_id}_{media_id}.{file_extension}"
        else:
            print(
                f"WARNING: no graphic is parsed from xml fig {media_id} in article {xml_path}."
            )
            continue

        if file_extension not in [
            "jpg",
            "png",
        ]:
            print(
                f"WARNING: {xml_path} contains media we don't know: {media_name}, {media_url}. Skipping."
            )
            continue

        # not all figs have captions, check PMC212690 as an example: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC212690/
        caption = "" if not fig.caption else fig.caption.get_text()

        item_info.append(
            {
                "PMC_ID": pmc_id,
                "media_id": media_id,  # media_id could represent image or video. [image: pbio-0020008-g002; video: pbio-0020008-v001]
                "caption": caption,
                "media_url": media_url,
                "media_name": media_name,
            }
        )

    return item_info


def get_volume_info(
    args: Namespace, volumes: List[int], extraction_dir: pathlib.Path
) -> List[Dict[str, str]]:
    """Extract <img, caption> pairs from given volumes of Pubmed articles.

    Parameters
    ----------
    args: argparse.Namespace
        Commandline arguments of the whole module.
    volumes: List[int]
        List of indices of volumes.
    extraction_dir: pathlib.Path
        Directory where extracted <img, caption> pairs will be stored.

    Returns
    -------
    info: List[Dict[str, str]]
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
    if not isinstance(extraction_dir, pathlib.Path):
        extraction_dir = pathlib.Path(extraction_dir)
    info = []
    for volume_id in volumes:
        volume = "PMC0%02dxxxxxx" % volume_id
        file_name = f"oa_{args.license_type}_xml.{volume}.baseline.{UPDATE_SCHEDULE}.filelist.csv"
        file_path = extraction_dir / volume / file_name

        df = pd.read_csv(file_path, sep=",")

        for idx in tqdm(range(len(df)), desc="parse xml"):
            xml_path = extraction_dir / volume / df.loc[idx, "Article File"]
            item_info = parse_xml(args, xml_path)
            info += item_info
    return info
