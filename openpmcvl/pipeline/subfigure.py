import os
import json
import argparse
from sys import stderr
from tqdm import tqdm
from typing import List, Dict, Any

import torch
from torchvision import transforms
from PIL import Image

from models.subfigure_detector import FigCap_Former


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each representing an item in the dataset.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f][23:25]


def process_image(image_path: str, transform: transforms.Compose) -> torch.Tensor:
    """
    Process an image for model input.

    Args:
        image_path (str): Path to the image file.
        transform: Image transformation pipeline.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def run_inference(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device, detection_threshold: float) -> Dict[str, Any]:
    """
    Run inference on an image using the FigCap_Former model.

    Args:
        model (torch.nn.Module): The FigCap_Former model.
        image_tensor (torch.Tensor): Processed image tensor.
        device (torch.device): Device to run inference on.
        detection_threshold (float): Threshold for positive detections.

    Returns:
        Dict[str, Any]: Dictionary containing number of subfigures and bounding boxes.
    """
    model.eval()

    with torch.no_grad():
        output_det_class, output_box, _ = model(image_tensor, None)
    
    positive_detections = output_det_class.squeeze() > detection_threshold
    detected_boxes = output_box.squeeze()[positive_detections].tolist()
    
    return {
        "num_subfigures": positive_detections.sum().item(),
        "bounding_boxes": detected_boxes
    }


def main(args: argparse.Namespace) -> None:
    """
    Main function to process images and save results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    model = FigCap_Former(
        num_query=32, num_encoder_layers=4, num_decoder_layers=4, num_text_decoder_layers=4,
        bert_path='bert-base-uncased', alignment_network=False, resnet_pretrained=False,
        resnet=34, feature_dim=256, atten_head_num=8, text_atten_head_num=8,
        mlp_ratio=4, dropout=0.0, activation='relu',
        text_mlp_ratio=4, text_dropout=0.0, text_activation='relu'
    )
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)

    # Set up image transformation
    image_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = load_dataset(args.input_file)
    print(f"\nDataset size: {len(dataset)}")

    # Process images
    results = []

    for item in tqdm(dataset, desc="Processing images", total=len(dataset), file=stderr):
        image_path = os.path.join(args.data_root, "figures", item['media_name'], item['media_url'].split('/')[-1])
        image_tensor = process_image(image_path, image_transform).to(device)
        
        inference_result = run_inference(model, image_tensor, device, args.detection_threshold)
        
        item.update(inference_result)
        results.append(item)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to detect subfigures")
    
    parser.add_argument("--input-file", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-file", required=True, help="Path to output JSON file")
    parser.add_argument("--data-root", default="/datasets/PMC-15M", help="Root directory for dataset")
    parser.add_argument("--model-path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--detection-threshold", type=float, default=0.6, help="Threshold for positive detections")
    
    args = parser.parse_args()
    main(args)
