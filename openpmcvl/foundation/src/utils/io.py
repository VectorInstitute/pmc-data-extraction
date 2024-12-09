"""Read and write to jsonl."""

from typing import Any, List

import jsonlines


def read_jsonl(file_path: str) -> List[Any]:
    """Read a list of objects (usually dict) from jsonl file."""
    data_list = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data_list.append(obj)
    return data_list


def write_jsonl(data_list: List[Any], save_path: str) -> None:
    """Write a list of objects (usually dict) to jsonl file."""
    with jsonlines.open(save_path, mode="w") as writer:
        for data in data_list:
            writer.write(data)
