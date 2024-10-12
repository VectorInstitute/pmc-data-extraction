import argparse
import torch
from typing import List, Dict
from open_clip import create_model_from_pretrained, get_tokenizer


def load_biomedclip_model(device: torch.device) -> tuple:
    """
    Load the BiomedCLIP model and tokenizer.

    Args:
        device (torch.device): The device to load the model on.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def get_keyword_embeddings(
    model: torch.nn.Module, tokenizer: object, classes: List[str], device: torch.device
) -> torch.Tensor:
    """
    Generate embeddings for the given classes using BiomedCLIP.

    Args:
        model (torch.nn.Module): The BiomedCLIP model.
        tokenizer (object): The BiomedCLIP tokenizer.
        classes (List[str]): List of classes to embed.
        device (torch.device): The device to perform computations on.

    Returns:
        torch.Tensor: Tensor of keyword embeddings.
    """
    template = "this is a photo of "
    texts = tokenizer([template + k for k in classes], context_length=256).to(device)
    with torch.no_grad():
        _, text_features, _ = model(None, texts)
    return text_features


def classify_images(
    image_embeddings: torch.Tensor, keyword_embeddings: torch.Tensor
) -> torch.Tensor:
    """
    Classify images based on the closest keyword embedding.

    Args:
        image_embeddings (torch.Tensor): Tensor of image embeddings.
        keyword_embeddings (torch.Tensor): Tensor of keyword embeddings.

    Returns:
        torch.Tensor: Tensor of classification indices.
    """
    similarities = torch.matmul(image_embeddings, keyword_embeddings.t())
    return torch.argmax(similarities, dim=1)


def main(input_file: str, output_file: str, classes: List[str]):
    """
    Main function to classify images using BiomedCLIP.

    Args:
        input_file (str): Path to the input .pt file containing image embeddings.
        output_file (str): Path to save the output .pt file with classifications.
        classes (List[str]): List of classes to use for classification.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_biomedclip_model(device)

    keyword_embeddings = get_keyword_embeddings(model, tokenizer, classes, device)

    input_data = torch.load(input_file)
    image_embeddings = input_data["rgb_embedding"].to(device)

    classifications = classify_images(image_embeddings, keyword_embeddings)

    input_data["class"] = classifications.cpu()

    torch.save(input_data, output_file)
    print(f"Classifications saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images using BiomedCLIP")
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .pt file containing image embeddings",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the output .pt file with classifications",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["radiology", "pathology", "chest x-ray"],
        help="List of classes to use for classification",
    )

    args = parser.parse_args()

    main(args.input_file, args.output_file, args.classes)
