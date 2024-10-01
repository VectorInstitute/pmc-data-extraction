import json
import os
import re

from tqdm import tqdm


def file_captions(jsonl_file):
    # create output dir
    cap_rootdir = os.path.join(os.path.dirname(jsonl_file), "captions")  # captions root dir
    if not os.path.isdir(cap_rootdir):
        os.mkdir(cap_rootdir)

    # load jsonl file
    print(f"Loading data from {jsonl_file} ...")
    with open(jsonl_file, "r") as f:
        data = [json.loads(line) for line in f]

    for sample in tqdm(data):
        # write caption in a separate file
        cap_filename = f"{sample['PMC_ID']}_{sample['media_id']}.txt"
        with open(os.path.join(cap_rootdir, cap_filename), "w") as f:
            f.write(sample["caption"])

        # replace caption with caption file name in jsonl
        sample.pop("caption", None)
        sample["caption_name"] = cap_filename

    # save new entries file
    with open(jsonl_file.replace(".jsonl", "_.jsonl"), "w") as f:
        for sample in data:
            json.dump(sample, f)
            f.write("\n")


if __name__ == "__main__":
    jsonl_file = "/datasets/PMC-15M/processed/train.jsonl"
    file_captions(jsonl_file)
    jsonl_file = "/datasets/PMC-15M/processed/test.jsonl"
    file_captions(jsonl_file)

