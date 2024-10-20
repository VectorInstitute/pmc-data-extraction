"""Convert PMC-OA data files to csv readable by open_clip."""
import json
import os
import pandas as pd


def convert_jsonl_to_csv(input_file: str, output_file: str, img_key: str = "image", cap_key: str = "caption") -> None:
    """Convert Jsonl entries to Csv entries."""
    with open(input_file, "r") as file:
        entries = [json.loads(line) for line in file]
    entries_df = pd.DataFrame.from_records(entries)
    root_dir = os.path.dirname(input_file)
    entries_df["image"] = entries_df["image"].apply(lambda x: os.path.join(root_dir, "images", x))
    entries_df.to_csv(output_file, sep=",")
    print(f"Data stored in {output_file}")


if __name__ == "__main__":
    for split in ["test", "valid", "train"]:
        print(f"Converting {split} split...")
        input_file = os.path.join(os.getenv("PMCOA_ROOT_DIR"), f"{split}.jsonl")
        output_file = os.path.join(os.getenv("PMCOA_ROOT_DIR"), f"{split}.csv")
        convert_jsonl_to_csv(input_file, output_file)
