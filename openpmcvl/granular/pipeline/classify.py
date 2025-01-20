import argparse
from typing import Any, Dict, List

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from openpmcvl.granular.dataset.dataset import SubfigureDataset
from openpmcvl.granular.pipeline.utils import load_dataset, save_jsonl


MEDICAL_CLASS = 15
CLASSIFICATION_THRESHOLD = 4


def load_classification_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads the figure classification model.

    Args:
        model_path (str): Path to the classification model checkpoint
        device (torch.device): Device to use for processing

    Returns
    -------
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
    num_workers: int,
):
    """
    Classifies images in a dataset using the provided model and saves results to a new JSONL file.

    Args:
        model (torch.nn.Module): The classification model.
        data_list (List[Dict[str, Any]]): The dataset to classify.
        batch_size (int): Batch size for processing.
        device (torch.device): Device to use for processing.
        output_file (str): Path to save the updated JSONL file with classification results.
        num_workers (int): Number of workers for processing.

    Returns
    -------
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model.eval()
    model.to(device)

    results = []

    for images, indices in tqdm(
        dataloader, desc=f"Classifying for {output_file}", total=len(dataloader)
    ):
        images = images.to(device)
        outputs = model(images)

        for output, idx in zip(outputs, indices):
            sorted_pred = torch.argsort(output.cpu(), descending=True)
            medical_class_rank = (sorted_pred == MEDICAL_CLASS).nonzero().item()
            is_medical = medical_class_rank < CLASSIFICATION_THRESHOLD

            # Get the original item using the index
            item = data_list[idx.item()]
            result = {
                **item,
                "medical_class_rank": medical_class_rank,
                "is_medical_subfigure": is_medical,
            }
            results.append(result)

    save_jsonl(results, output_file)


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

    Returns
    -------
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    model = load_classification_model(args.model_path, device)
    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} subfigures from {args.dataset_path}.")

    classify_dataset(
        model,
        dataset,
        args.batch_size,
        device,
        args.output_file,
        args.num_workers,
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
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for processing"
    )
    args = parser.parse_args()

    main(args)
