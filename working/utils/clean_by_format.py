import json
import os

from PIL import Image
from tqdm import tqdm


def clean_by_format(jsonl_file, output_file, accepted_ext):
    # load jsonl file
    print(f"Loading data from {jsonl_file} ...")
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f]

    clean_data = []
    for sample in tqdm(data):
        # check media format
        if sample["media_name"].split(".")[-1] in accepted_ext:
            clean_data.append(sample)


    # save new entries file
    print(f"Saving data to {output_file} ...")
    with open(output_file, "w") as f:
        for sample in clean_data:
            json.dump(sample, f)
            f.write("\n")


def check_loadability(jsonl_file, output_file):
    # load jsonl file
    print(f"Loading data from {jsonl_file} ...")
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f]

    num_error = 0
    clean_data = []
    for sample in tqdm(data):
        # try loading the image with PIL
        try:
            img_path = os.path.join(os.path.dirname(jsonl_file), "figures", sample["media_name"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")
            # if no exception occured, sample is loadable
            clean_data.append(sample)
        except Exception as e:
            print(f"Error loading {img_path}")
            print(e)
            num_error += 1
            continue

    print(f"Number of error file: {num_error}")

    # save new entries file
    print(f"Saving data to {output_file} ...")
    with open(output_file, "w") as f:
        for sample in clean_data:
            json.dump(sample, f)
            f.write("\n")


if __name__ == "__main__":
    jsonl_file = "/datasets/PMC-15M/processed/train_.jsonl"
    output_file = "/datasets/PMC-15M/processed/train_clean.jsonl"
    accepted_ext = ["jpg", "png", "jpeg", "blp", "bmp", "gif", "dds", "dib", "eps"]
    clean_by_format(jsonl_file, output_file, accepted_ext)

    checked_file = "/datasets/PMC-15M/processed/train_checked.jsonl"
    check_loadability(output_file, checked_file)

