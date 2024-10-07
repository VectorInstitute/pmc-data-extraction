import json
import os
import pickle
from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer

# store the start time
stime = time()

# get file names
rootdir = "/datasets/PMC-15M/"
filenames = [fname for fname in os.listdir(rootdir) if fname.endswith(".jsonl") and not fname.endswith("_refs.jsonl")]
filenames = sorted(filenames, key=lambda x: int(x.replace(".jsonl", "")))

print(f"[{timedelta(seconds=int(time() - stime))}] Name of found files in {rootdir}:")
print("\t"+"\n\t".join(filenames))

# create results directory
if not os.path.exists(os.path.join(rootdir, "text_lengths")):
    os.mkdir(os.path.join(rootdir, "text_lengths"))
print(f"[{timedelta(seconds=int(time() - stime))}] Results directory is set to {os.path.join(rootdir, 'text_lengths')}")

# load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

for fname in filenames:
    text_lengths = []
    print(f"[{timedelta(seconds=int(time() - stime))}] Loading file {fname}...")
    # load data
    with open(os.path.join(rootdir, fname)) as f:
        data = [json.loads(line)["caption"] for line in f]

    print(f"[{timedelta(seconds=int(time() - stime))}] Extracting caption lengths in {fname}...")
    for caption in data:
        tokens = tokenizer(caption, return_tensors='pt')
        text_lengths.append(tokens["input_ids"].reshape(-1).shape[0])

    print(f"[{timedelta(seconds=int(time() - stime))}] Saving caption lengths of {fname}...")
    save_path = os.path.join(rootdir, "text_lengths", fname.replace(".jsonl", ".pkl"))
    with open(save_path, "wb") as f:
        pickle.dump(text_lengths, f)
    print(f"[{timedelta(seconds=int(time() - stime))}] Saved caption lengths in {save_path}...")
print(f"[{timedelta(seconds=int(time() - stime))}] Successfully processed all files in {rootdir}")





