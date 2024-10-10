import os
import re
import argparse
from typing import List, Tuple
from tqdm import tqdm

from PIL import Image
from openpmcvl.granular.pipeline.utils import load_dataset, save_jsonl


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get the width and height of an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Tuple[int, int]: A tuple containing the width and height of the image.

    Raises:
        IOError: If the image file cannot be opened or read.
    """
    with Image.open(image_path) as img:
        return img.size


def check_keywords(caption: str, keywords: List[str]) -> Tuple[List[str], bool]:
    """
    Check for the presence of keywords in the caption.

    Args:
        caption (str): The caption text to search in.
        keywords (List[str]): A list of keywords to search for.

    Returns:
        Tuple[List[str], bool]: A tuple containing:
            - A list of found keywords.
            - A boolean indicating whether any keywords were found.
    """
    found_keywords = [
        kw
        for kw in keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", caption, re.IGNORECASE)
    ]
    return found_keywords, bool(found_keywords)


def preprocess_data(
    input_files: List[str], output_file: str, figure_root: str, keywords: List[str]
) -> None:
    """
    Preprocess the input files and generate the output file.

    Args:
        input_files (List[str]): List of input JSONL file paths.
        output_file (str): Path to the output JSONL file.
        figure_root (str): Root directory for figure images.
        keywords (List[str]): List of keywords to search for in captions.

    Returns:
        None

    Raises:
        IOError: If there are issues reading input files or writing the output file.
    """
    processed_data = []

    for input_file in tqdm(input_files, desc="Processing files"):
        data = load_dataset(input_file, num_datapoints=-1)
        for item in tqdm(data, desc=f"Processing items in {input_file}", leave=False):
            pmc_id = item["PMC_ID"]
            media_id = item["media_id"]
            image_name = os.path.basename(item["media_url"])
            image_path = f"{figure_root}/{pmc_id}_{media_id}.jpg/{image_name}"

            # Skip if image doesn't exist
            if not os.path.exists(image_path):
                if image_path.endswith(".jpg"):
                    print(f"Image not found: {image_path}")
                continue

            try:
                width, height = get_image_dimensions(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue

            found_keywords, has_keywords = check_keywords(item["caption"], keywords)

            processed_item = {
                "id": f"{pmc_id}_{media_id}",
                "PMC_ID": pmc_id,
                "caption": item["caption"],
                "image_path": image_path,
                "width": width,
                "height": height,
                "media_id": media_id,
                "media_url": item["media_url"],
                "media_name": item["media_name"],
                "keywords": found_keywords,
                "is_medical": has_keywords,
            }
            processed_data.append(processed_item)

    save_jsonl(processed_data, output_file)
    print(f"Processed data saved to {output_file}")


def main():
    """
    Main function to parse arguments and run the preprocessing pipeline.

    Args:
        None

    Returns:
        None

    Raises:
        ArgumentError: If required arguments are missing or invalid.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess JSONL files for figure caption analysis"
    )
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        required=True,
        help="List of input JSONL files",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to the output JSONL file"
    )
    parser.add_argument(
        "--figure_root",
        type=str,
        required=True,
        help="Root directory for figure images",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=["CT", "pathology", "radiology"],
        help="Keywords to search for in captions",
    )

    args = parser.parse_args()

    preprocess_data(args.input_files, args.output_file, args.figure_root, args.keywords)


if __name__ == "__main__":
    main()
