import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from tqdm import tqdm

from openpmcvl.granular.dataset.dataset import (
    Fig_Separation_Dataset,
    fig_separation_collate,
)
from openpmcvl.granular.models.subfigure_detector import FigCap_Former
from openpmcvl.granular.pipeline.utils import (
    box_cxcywh_to_xyxy,
    find_jaccard_overlap,
    save_jsonl,
)


def load_dataset(eval_file: str, batch_size: int, num_workers: int) -> DataLoader:
    """
    Prepares the dataset and returns a DataLoader.

    Args:
        eval_file (str): Path to the evaluation dataset file
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for the DataLoader

    Returns
    -------
        DataLoader: Configured DataLoader for the separation dataset
    """
    dataset = Fig_Separation_Dataset(
        filepath=eval_file, normalization=False, only_medical=True
    )
    print(f"\nDataset size: {len(dataset)}\n")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=fig_separation_collate,
        pin_memory=True,
    )


def load_separation_model(checkpoint_path: str, device: torch.device) -> FigCap_Former:
    """
    Loads the FigCap_Former model from a checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to use for processing

    Returns
    -------
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

    Returns
    -------
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


def separate_subfigures(
    model: FigCap_Former,
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
        loader (DataLoader): DataLoader for the dataset
        save_path (str): Path to save separated subfigures
        rcd_file (str): File to record separation results
        score_threshold (float): Confidence score threshold for detections
        nms_threshold (float): IoU threshold for NMS
        device (torch.device): Device to use for processing
    """
    Path(save_path).mkdir(parents=True, exist_ok=True)
    subfig_list = []
    failed_subfig_list = []
    subfig_count = 0

    print("Separating subfigures...")
    for batch in tqdm(loader, desc=f"File: {rcd_file}", total=len(loader)):
        image = batch["image"].to(device)
        img_ids = batch["image_id"]
        original_images = batch["original_image"]
        unpadded_hws = batch["unpadded_hws"]

        output_det_class, output_box, _ = model(image, None)

        output_box = output_box.cpu()
        output_det_class = output_det_class.cpu()
        filter_mask = output_det_class.squeeze() > score_threshold

        for i in range(image.shape[0]):
            det_boxes = output_box[i, filter_mask[i, :], :]
            det_scores = output_det_class.squeeze()[i, filter_mask[i, :]].numpy()
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

                    # Calculate padding in terms of bounding box dimensions
                    pad_ratio = 0.01
                    pad_w = w * pad_ratio
                    pad_h = h * pad_ratio

                    # Adjust the coordinates with padding
                    x1 = round((cx - w / 2 - pad_w) * image.shape[3] * scale)
                    x2 = round((cx + w / 2 + pad_w) * image.shape[3] * scale)
                    y1 = round((cy - h / 2 - pad_h) * image.shape[2] * scale)
                    y2 = round((cy + h / 2 + pad_h) * image.shape[2] * scale)

                    # Ensure the coordinates are within image boundaries
                    x1, x2 = [max(0, min(x, original_w - 1)) for x in [x1, x2]]
                    y1, y2 = [max(0, min(y, original_h - 1)) for y in [y1, y2]]

                    subfig = unpadded_image[:, y1:y2, x1:x2].detach().cpu()
                    vutils.save_image(subfig, subfig_path)

                    subfig_list.append(
                        {
                            "id": f"{img_id}_{subfig_count}.jpg",
                            "source_fig_id": img_id,
                            "PMC_ID": img_id.split("_")[0],
                            "media_name": f"{img_id}.jpg",
                            "position": [(x1, y1), (x2, y2)],
                            "score": score.item(),
                            "subfig_path": subfig_path,
                        }
                    )
                    subfig_count += 1
                except ValueError:
                    print(
                        f"Crop Error: [x1 x2 y1 y2]:[{x1} {x2} {y1} {y2}], w:{original_w}, h:{original_h}"
                    )
                    failed_subfig_list.append(
                        {
                            "id": f"{img_id}_{subfig_count}.jpg",
                            "source_fig_id": img_id,
                            "PMC_ID": img_id.split("_")[0],
                            "media_name": f"{img_id}.jpg",
                        }
                    )
                    continue

    save_jsonl(subfig_list, rcd_file)
    save_jsonl(failed_subfig_list, f"{rcd_file.split('.')[0]}_failed.jsonl")


def main(args: argparse.Namespace) -> None:
    """
    Main function to process images and save results.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    model = load_separation_model(args.separation_model, device)
    dataloader = load_dataset(args.eval_file, args.batch_size, args.num_workers)
    separate_subfigures(
        model=model,
        loader=dataloader,
        save_path=args.save_path,
        rcd_file=args.rcd_file,
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        device=device,
    )
    print("\nSubfigure separation completed.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subfigure Separation Script")

    parser.add_argument(
        "--separation_model",
        type=str,
        required=True,
        help="Path to subfigure detection model checkpoint",
    )
    parser.add_argument(
        "--eval_file", type=str, required=True, help="Path to evaluation dataset file"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save separated subfigures",
    )
    parser.add_argument(
        "--rcd_file",
        type=str,
        required=True,
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
        "--batch_size", type=int, default=128, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of workers for data loading"
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use")

    args = parser.parse_args()
    main(args)
