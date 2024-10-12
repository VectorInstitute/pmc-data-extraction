import json
from typing import Any, Dict, List


def load_dataset(file_path: str, num_datapoints: int = -1) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        file_path (str): Path to the input JSONL file.
        num_datapoints (int): Number of datapoints to load. If -1, load all datapoints.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each representing an item in the dataset.
    """
    with open(file_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    return dataset[:num_datapoints] if num_datapoints > 0 else dataset


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    with open(file_path, "w") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")
