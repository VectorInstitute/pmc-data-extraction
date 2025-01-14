import json
from typing import Any, Dict, List

import torch


def load_dataset(file_path: str, num_datapoints: int = -1) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file.
        num_datapoints (int): Number of datapoints to load. If -1, load all datapoints.

    Returns
    -------
        List[Dict[str, Any]]: List of dictionaries, each representing an item in the dataset.
    """
    with open(file_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    return dataset[:num_datapoints] if num_datapoints > 0 else dataset


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL (JSON Lines) file.

    This function takes a list of dictionaries and writes each dictionary as a separate JSON object
    on a new line in the specified file. This format is known as JSONL (JSON Lines).

    Args:
        data (List[Dict[str, Any]]): A list of dictionaries to be saved. Each dictionary
                                     represents a single data point or record.
        file_path (str): The path to the output file where the data will be saved.
    """
    with open(file_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box coordinates from (center_x, center_y, width, height) to (x1, y1, x2, y2) format.

    Args:
        x (torch.Tensor): Input tensor of shape (..., 4) containing bounding box coordinates in (cx, cy, w, h) format.

    Returns
    -------
        torch.Tensor: Tensor of shape (..., 4) containing bounding box coordinates in (x1, y1, x2, y2) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1 (torch.Tensor): Set 1, a tensor of dimensions (n1, 4) -- (x1, y1, x2, y2)
        set_2 (torch.Tensor): Set 2, a tensor of dimensions (n2, 4)

    Returns
    -------
        torch.Tensor: Intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    Args:
        set_1 (torch.Tensor): Set 1, a tensor of dimensions (n1, 4)
        set_2 (torch.Tensor): Set 2, a tensor of dimensions (n2, 4)

    Returns
    -------
        torch.Tensor: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """
    if set_1.dim() == 1 and set_1.shape[0] == 4:
        set_1 = set_1.unsqueeze(0)
    if set_2.dim() == 1 and set_2.shape[0] == 4:
        set_2 = set_2.unsqueeze(0)

    intersection = find_intersection(set_1, set_2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection

    return intersection / union
