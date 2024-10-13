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
    caption_lower = caption.lower()
    found_keywords = [
        kw for kw in keywords if f" {kw.lower()} " in f" {caption_lower} "
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
    all_processed_data = []
    missing_figures = {}
    output_dir = os.path.dirname(output_file)

    for input_file in input_files:
        data = load_dataset(input_file, num_datapoints=-1)
        processed_data = []
        missing_figures_count = 0

        for item in tqdm(data, desc=f"Processing items in {input_file}", leave=False):
            pmc_id = item["PMC_ID"]
            media_id = item["media_id"]
            media_name = item["media_name"]
            media_url = item["media_url"]
            caption = item["caption"]
            image_name = os.path.basename(media_url)
            image_path = f"{figure_root}/{media_name}/{image_name}"

            # Check if image doesn't exist or is not a .jpg file
            if not image_path.endswith(".jpg"):
                continue

            if not os.path.exists(image_path):
                missing_figures_count += 1
                if input_file not in missing_figures:
                    missing_figures[input_file] = []
                missing_figures[input_file].append(image_path)
                continue

            try:
                width, height = get_image_dimensions(image_path)
            except Exception as e:
                print(f"Error processing image {image_path}: {str(e)}")
                continue

            found_keywords, has_keywords = check_keywords(caption, keywords)

            processed_item = {
                "id": f"{pmc_id}_{media_id}",
                "PMC_ID": pmc_id,
                "caption": caption,
                "image_path": image_path,
                "width": width,
                "height": height,
                "media_id": media_id,
                "media_url": media_url,
                "media_name": media_name,
                "keywords": found_keywords,
                "is_medical": has_keywords,
            }
            processed_data.append(processed_item)

        # Save processed data for this input file
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        temp_output_file = os.path.join(output_dir, f"{input_filename}_processed.jsonl")
        save_jsonl(processed_data, temp_output_file)
        print(
            f"\nProcessed {len(processed_data)} items from {input_file}. Saved to {temp_output_file}"
            f"\nTotal missing .jpg images in {input_file}: {missing_figures_count}\n"
        )

        all_processed_data.extend(processed_data)

    # Merge all processed data and save to the final output file
    save_jsonl(all_processed_data, output_file)
    print(f"All processed data merged and saved to {output_file}")

    # Save missing images to a separate JSONL file
    missing_figures_file = os.path.join(output_dir, "missing_figures.jsonl")
    save_jsonl(missing_figures, missing_figures_file)
    print(f"Missing images information saved to {missing_figures_file}")


def main(args: argparse.Namespace):
    """
    Main function to parse arguments and run the preprocessing pipeline.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    preprocess_data(args.input_files, args.output_file, args.figure_root, args.keywords)


if __name__ == "__main__":
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
    main(args)
