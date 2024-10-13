import argparse
import json
from PIL import Image
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from openpmcvl.granular.pipeline.utils import load_dataset
from openpmcvl.granular.dataset.dataset import SubfigureDataset

MEDICAL_CLASS = 15
CLASSIFICATION_THRESHOLD = 4


def load_classification_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads the figure classification model.

    Args:
        model_path (str): Path to the classification model checkpoint
        device (torch.device): Device to use for processing

    Returns:
        nn.Module: Loaded classification model
    """
    fig_model = models.resnext101_32x8d()
    num_features = fig_model.fc.in_features
    fc = list(fig_model.fc.children())
    fc.extend([nn.Linear(num_features, 28)])
    fig_model.fc = nn.Sequential(*fc)
    fig_model = fig_model.to(device)
    fig_model.load_state_dict(torch.load(model_path, map_location=device))
    fig_model.eval()
    return fig_model


def classify_dataset(
    model: torch.nn.Module,
    data_list: List[Dict[str, Any]],
    batch_size: int,
    device: torch.device,
    output_file: str,
):
    """
    Classifies images in a dataset using the provided model and saves results to a new JSONL file.

    Args:
        model (torch.nn.Module): The classification model.
        data_list (List[Dict[str, Any]]): The dataset to classify.
        batch_size (int): Batch size for processing.
        device (torch.device): Device to use for processing.
        output_file (str): Path to save the updated JSONL file with classification results.

    Returns:
        None
    """
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
        ]
    )

    dataset = SubfigureDataset(data_list, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    model.to(device)

    results = []

    with torch.no_grad():
        for images, items in tqdm(dataloader, desc="Classifying"):
            images = images.to(device)

            outputs = model(images)

            for output, item in zip(outputs, items):
                sorted_pred = torch.argsort(output.cpu(), descending=True)
                medical_class_rank = (sorted_pred == MEDICAL_CLASS).nonzero().item()
                is_medical = medical_class_rank < CLASSIFICATION_THRESHOLD

                # Update the item with the new keys
                item["medical_class_rank"] = medical_class_rank
                item["is_medical"] = is_medical

                # Append the updated item to results
                results.append(item)

    # Save the updated items to a new JSONL file
    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the image classification pipeline.

    This function loads the classification model, prepares the dataset,
    and performs classification on the images, saving the results to a JSONL file.

    Args:
        args (argparse.Namespace): Command-line arguments containing:
            - model_path (str): Path to the classification model checkpoint
            - dataset_path (str): Path to the dataset
            - batch_size (int): Batch size for processing
            - output_file (str): Path to save the JSONL file with classification results

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_classification_model(args.model_path, device)
    dataset = load_dataset(args.dataset_path)

    classify_dataset(
        model, dataset, args.batch_size, device, args.dataset_path, args.output_file
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify images in a dataset and update JSONL file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the classification model checkpoint",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the JSONL file with classification results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for processing"
    )
    args = parser.parse_args()

    main(args)
