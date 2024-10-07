import argparse
import json
from pathlib import Path
from typing import List, Dict, Union

from tqdm import tqdm
from openpmcvl.granular.process.playground_subfigure_ocr import classifier


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        List[Dict]: List of dictionaries containing the JSONL data
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def update_dataset(data: List[Dict], file_path: str) -> None:
    """
    Save dataset to a JSONL file.

    Args:
        data (List[Dict]): List of dictionaries to save
        file_path (str): Path to the output JSONL file
    """
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')

def process_subfigure(model: classifier, subfig_data: Dict) -> Dict:
    """
    Process a single subfigure using the OCR model.

    Args:
        model (classifier): Initialized OCR model
        subfig_data (Dict): Dictionary containing subfigure data

    Returns:
        Dict: Updated subfigure data with OCR results
    """
    image_path = subfig_data['subfig_path']
    ocr_result = model.run(image_path)

    if ocr_result:
        label_letter, *label_position = ocr_result
        subfig_data['label'] = f"Subfigure-{label_letter.upper()}"
        subfig_data['label_position'] = label_position
    else:
        subfig_data['label'] = ""
        subfig_data['label_position'] = []

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

    # Process each subfigure
    updated_data = []
    for item in tqdm(dataset, desc="Processing subfigures", total=len(dataset)):
        updated_item = process_subfigure(model, item)
        updated_data.append(updated_item)

    # Save updated data
    update_dataset(updated_data, args.save_path)
    print(f"\nUpdated data saved to {args.save_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subfigure OCR and Labeling")

    parser.add_argument('--dataset_path', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to output JSONL file')
    
    args = parser.parse_args()
    main(args)