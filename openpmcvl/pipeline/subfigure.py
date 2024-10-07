import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import utils as vutils, models, transforms
from PIL import Image

from openpmcvl.process.dataset_det_align import (
    Fig_Separation_Dataset,
    fig_separation_collate,
)
from openpmcvl.models.subfigure_detector import FigCap_Former
from openpmcvl.process.detect_metric import box_cxcywh_to_xyxy, find_jaccard_overlap

MEDICAL_CLASS = 15
CLASSIFICATION_THRESHOLD = 4


def load_dataset(
    eval_file: str, img_root: str, batch_size: int, num_workers: int
) -> DataLoader:
    """
    Prepares the dataset and returns a DataLoader.

    Args:
        eval_file (str): Path to the evaluation dataset file
        img_root (str): Root path for figures
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for the DataLoader

    Returns:
        DataLoader: Configured DataLoader for the separation dataset
    """
    dataset = Fig_Separation_Dataset(None, eval_file, img_root, normalization=False)
    print(f"\nDataset size: {len(dataset)}\n")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fig_separation_collate,
    )


def load_separation_model(checkpoint_path: str, device: torch.device) -> FigCap_Former:
    """
    Loads the FigCap_Former model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to use for processing

    Returns:
        FigCap_Former: Loaded model
    """
    model = FigCap_Former(
        num_query=32,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_text_decoder_layers=4,
        bert_path="bert-base-uncased",
        alignment_network=False,
        resnet_pretrained=False,
        resnet=34,
        feature_dim=256,
        atten_head_num=8,
        text_atten_head_num=8,
        mlp_ratio=4,
        dropout=0.0,
        activation="relu",
        text_mlp_ratio=4,
        text_dropout=0.0,
        text_activation="relu",
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.eval()
    model.to(device)

    return model


def process_detections(
    det_boxes: torch.Tensor, det_scores: np.ndarray, nms_threshold: float
) -> Tuple[List[List[float]], List[float]]:
    """
    Processes detections using Non-Maximum Suppression (NMS).

    Args:
        det_boxes (torch.Tensor): Detected bounding boxes
        det_scores (np.ndarray): Confidence scores for detections
        nms_threshold (float): IoU threshold for NMS

    Returns:
        Tuple[List[List[float]], List[float]]: Picked bounding boxes and their scores
    """
    order = np.argsort(det_scores)
    picked_bboxes = []
    picked_scores = []
    while order.size > 0:
        index = order[-1]
        picked_bboxes.append(det_boxes[index].tolist())
        picked_scores.append(det_scores[index])
        if order.size == 1:
            break
        iou_with_left = (
            find_jaccard_overlap(
                box_cxcywh_to_xyxy(det_boxes[index]),
                box_cxcywh_to_xyxy(det_boxes[order[:-1]]),
            )
            .squeeze()
            .numpy()
        )
        left = np.where(iou_with_left < nms_threshold)
        order = order[left]
    return picked_bboxes, picked_scores


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


def separate_classify_subfigures(
    model: FigCap_Former,
    class_model: nn.Module,
    loader: DataLoader,
    save_path: str,
    rcd_file: str,
    score_threshold: float,
    nms_threshold: float,
    device: torch.device,
) -> None:
    """
    Separates compound figures into subfigures and classifies them.

    Args:
        model (FigCap_Former): Loaded model for subfigure detection
        class_model (nn.Module): Loaded model for figure classification
        loader (DataLoader): DataLoader for the dataset
        save_path (str): Path to save separated subfigures
        rcd_file (str): File to record separation results
        score_threshold (float): Confidence score threshold for detections
        nms_threshold (float): IoU threshold for NMS
        device (torch.device): Device to use for processing
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subfig_list = []
    subfig_count = 0

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    fig_class_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std),
        ]
    )

    with torch.no_grad():
        try:
            for batch in tqdm(loader, desc="Separating subfigures", total=len(loader)):
                image = batch["image"].to(device)
                caption = batch["caption"].to(device)
                img_ids = batch["image_id"]
                original_images = batch["original_image"]
                unpadded_hws = batch["unpadded_hws"]

                output_det_class, output_box, _ = model(image, caption)

                cpu_output_box = output_box.cpu()
                cpu_output_det_class = output_det_class.cpu()
                filter_mask = cpu_output_det_class.squeeze() > score_threshold

                for i in range(image.shape[0]):
                    det_boxes = cpu_output_box[i, filter_mask[i, :], :]
                    det_scores = cpu_output_det_class.squeeze()[
                        i, filter_mask[i, :]
                    ].numpy()
                    img_id = img_ids[i].split(".jpg")[0]
                    unpadded_image = original_images[i]
                    original_h, original_w = unpadded_hws[i]

                    scale = max(original_h, original_w) / 512

                    picked_bboxes, picked_scores = process_detections(
                        det_boxes, det_scores, nms_threshold
                    )

                    for bbox, score in zip(picked_bboxes, picked_scores):
                        try:
                            subfig_path = f"{save_path}/{img_id}_{subfig_count}.jpg"
                            cx, cy, w, h = bbox
                            x1, x2 = [
                                round((cx - w / 2) * image.shape[3] * scale),
                                round((cx + w / 2) * image.shape[3] * scale),
                            ]
                            y1, y2 = [
                                round((cy - h / 2) * image.shape[2] * scale),
                                round((cy + h / 2) * image.shape[2] * scale),
                            ]

                            x1, x2 = [max(0, min(x, original_w - 1)) for x in [x1, x2]]
                            y1, y2 = [max(0, min(y, original_h - 1)) for y in [y1, y2]]

                            subfig = unpadded_image[:, y1:y2, x1:x2].to(
                                torch.device("cpu")
                            )
                            vutils.save_image(subfig, subfig_path)

                            # Perform classification
                            class_input = (
                                fig_class_transform(transforms.ToPILImage()(subfig))
                                .unsqueeze(0)
                                .to(device)
                            )
                            fig_prediction = class_model(class_input)

                            sorted_pred = torch.argsort(
                                fig_prediction[0].cpu(), descending=True
                            )
                            medical_class_rank = (sorted_pred == MEDICAL_CLASS).nonzero().item()
                            is_medical  = medical_class_rank < CLASSIFICATION_THRESHOLD

                            subfig_list.append(
                                {
                                    "id": f"{subfig_count}.jpg",
                                    "source_fig_id": img_id,
                                    "position": [(x1, y1), (x2, y2)],
                                    "score": score.item(),
                                    "subfig_path": subfig_path,
                                    "medical_class_rank": medical_class_rank,
                                    "is_medical": is_medical,
                                }
                            )
                            subfig_count += 1
                        except ValueError:
                            print(
                                f"Crop Error: [x1 x2 y1 y2]:[{x1} {x2} {y1} {y2}], w:{original_w}, h:{original_h}"
                            )
        except Exception as e:
            print(f"Error occurred: {repr(e)}")
        finally:
            with open(rcd_file, "a") as f:
                for line in subfig_list:
                    f.write(json.dumps(line) + "\n")


def main(args: argparse.Namespace) -> None:
    """
    Main function to process images and save results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_separation_model(args.separation_model, device)
    class_model = load_classification_model(args.class_model, device)
    dataloader = load_dataset(
        args.eval_file, args.img_root, args.batch_size, args.num_workers
    )
    separate_classify_subfigures(
        model,
        class_model,
        dataloader,
        args.save_path,
        args.rcd_file,
        args.score_threshold,
        args.nms_threshold,
        device,
    )
    print("\nSubfigure separation and classification completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subfigure Separation and Classification Script"
    )

    parser.add_argument(
        "--separation_model",
        type=str,
        required=True,
        help="Path to subfigure detection model checkpoint",
    )
    parser.add_argument(
        "--class_model",
        type=str,
        required=True,
        help="Path to figure classification model checkpoint",
    )
    parser.add_argument(
        "--eval_file", type=str, required=True, help="Path to evaluation dataset file"
    )
    parser.add_argument(
        "--img_root", type=str, required=True, help="Root path for figures"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./Separation",
        help="Path to save separated subfigures",
    )
    parser.add_argument(
        "--rcd_file",
        type=str,
        default="./Separation/separation.jsonl",
        help="File to record separation results",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.75,
        help="Confidence score threshold for detections",
    )
    parser.add_argument(
        "--nms_threshold", type=float, default=0.4, help="IoU threshold for NMS"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of workers for data loading"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

    args = parser.parse_args()
    main(args)
