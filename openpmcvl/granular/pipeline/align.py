import argparse
import os
from typing import Dict

from tqdm import tqdm

from openpmcvl.granular.models.subfigure_ocr import classifier
from openpmcvl.granular.pipeline.utils import load_dataset, save_jsonl


def process_subfigure(model: classifier, subfig_data: Dict) -> Dict:
    """
    Process a single subfigure using the OCR model.

    Args:
        model (classifier): Initialized OCR model
        subfig_data (Dict): Dictionary containing subfigure data

    Returns
    -------
        Dict: Updated subfigure data with OCR results
    """
    if "subfig_path" not in subfig_data:
        subfig_data["subfig_path"] = f"{args.root_dir}/images/{subfig_data['image']}"

    try:
        ocr_result = model.run(subfig_data["subfig_path"])
    except Exception as e:
        ocr_result = ""
        print(f"Error processing subfigure {subfig_data['image']}: {e}")

    if ocr_result:
        label_letter, *label_position = ocr_result
        subfig_data["label"] = f"Subfigure-{label_letter.upper()}"
        subfig_data["label_position"] = label_position
    else:
        subfig_data["label"] = ""
        subfig_data["label_position"] = []

    return subfig_data


def main(args: argparse.Namespace) -> None:
    """
    Main function to process subfigures and update JSONL file.

    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # Load model and dataset
    model = classifier()
    dataset = load_dataset(args.dataset_path)
    if args.dataset_slice:
        dataset = dataset[args.dataset_slice]

    # Use this line to filter out non-medical subfigures if needed
    dataset = [data for data in dataset if data["is_medical_subfigure"]]
    print(
        f"Total {len(dataset)} medical subfigures from {os.path.basename(args.dataset_path)}"
    )

    # Label each subfigure
    labeled_dataset = []
    for data in tqdm(dataset, desc="Labeling subfigures", total=len(dataset)):
        updated_item = process_subfigure(model, data)
        labeled_dataset.append(updated_item)

    total_labeled = len([data for data in labeled_dataset if data["label"]])
    print(f"Total {total_labeled} subfigures labeled.")

    # Save updated data
    save_jsonl(labeled_dataset, args.save_path)
    print(f"\nLabeled data saved to {args.save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subfigure OCR and Labeling")

    parser.add_argument(
        "--root_dir", type=str, required=True, help="Path to root directory"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--dataset_slice",
        type=str,
        help="Start and end indices for dataset slice (e.g. '0:100')",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to output JSONL file"
    )

    args = parser.parse_args()
    if args.dataset_slice:
        start, end = map(int, args.dataset_slice.split(":"))
        args.dataset_slice = slice(start, end)

    main(args)
