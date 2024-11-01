import json
import os
import re

import numpy as np
from tqdm import tqdm


def aggregate_vols(jsonl_rootdir, banned_extentions):
    """
    Gather all volumes in one file, fix media paths, and remove non-image files.
    """
    # list all available volumes
    volumes = [file for file in os.listdir(jsonl_rootdir) if re.fullmatch("[0-9]+\.jsonl", file) is not None]
    volumes = sorted(volumes, key=lambda x: int(x.replace(".jsonl", "")))
    print(f"Found volumes: {volumes}")

    img_rootdir = os.path.join(jsonl_rootdir, "figures")  # images root dir
    all_data = []  # list for aggregated dataset

    for volume in volumes:
        # load volume
        print(f"Processing volume {volume} ...")
        with open(os.path.join(jsonl_rootdir, volume), "r") as f:
            data = [json.loads(line) for line in f]

        nbanned = 0  # count number of non-image media
        for sample in tqdm(data):
            media_name = os.path.join(img_rootdir, sample["media_name"])
            # check if media file/dir exists
            if not os.path.exists(media_name):
                continue
            # check if media is image
            extention = sample["media_name"].split(".")[-1]
            if extention in banned_extentions:
                nbanned += 1
                continue
            # check if media_name is a file or directory
            # if media_name is a directory, replace it with the name of the file inside it
            if os.path.isdir(media_name):
                dir_contents = os.listdir(media_name)
                assert len(dir_contents) == 1, f"More than a single file exist in {media_name}"
                media_name = os.path.join(media_name, dir_contents[0])
                assert os.path.isfile(media_name), f"Is a directory: {media_name}"
                sample["media_name"] = os.path.join(sample["media_name"], dir_contents[0])
            # add sample to list
            if os.path.isfile(media_name):
                all_data.append(sample)

        print(f"{nbanned} non-image samples were in volume {volume}")
        print(f"{len(all_data)} samples aggregated.")

    return all_data


def train_test_split(agg_filename, train_ratio, test_ratio=None, seed=42):
    """
    Split dataset into train and test sets.
    """
    # load dataset
    print(f"Loading aggregated data: {agg_filename} ...")
    with open(agg_filename, "r") as f:
        data = [json.loads(line) for line in f]
    
    # determine number of train and test samples
    print("Determining number of train and test samples...")
    ntrain = int(len(data) * train_ratio)
    if test_ratio is not None:
        ntest = int(len(data) * test_ratio)
    else:
        ntest = len(data) - ntrain
    
    # randomly permute data
    print("Permuting data samples...")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(data)

    # split dataset
    train_data = perm[:ntrain]
    test_data = perm[ntrain:(ntrain+ntest)]
    print("Finished splitting:")
    print(f"num train: {len(train_data)}")
    print(f"num test: {len(test_data)}")

    return train_data, test_data


def save_jsonl(data, filename):
    with open(filename, "w") as f:
        for sample in tqdm(data):
            json.dump(sample, f)
            f.write("\n")


def create_dummy(filename, nsamples=1000):
    # load dataset
    print(f"Loading original data: {filename} ...")
    with open(filename, "r") as f:
        data = [json.loads(line) for line in f]
    # save dummy version
    save_jsonl(data[:nsamples], filename.replace(".jsonl", "_dummy.jsonl"))


def aggregate_files(filename1, filename2, outputfilename):
    # load dataset
    print(f"Loading data: {filename1} ...")
    with open(filename1, "r") as f:
        data1 = [json.loads(line) for line in f]
    
    # load dataset
    print(f"Loading data: {filename2} ...")
    with open(filename2, "r") as f:
        data2 = [json.loads(line) for line in f]
    
    print("Aggregating the two files")
    data1.extend(data2)

    print(f"Saving to new file: {outputfilename}")
    save_jsonl(data1, outputfilename)



if __name__ == "__main__":
    jsonl_rootdir = "/datasets/PMC-15M/processed"
    # banned_extentions = ["mov", "avi", "mpeg", "pdf", "mp4", "docx"]
    # agg_filename = os.path.join(jsonl_rootdir, "aggregated.jsonl")

    # # aggregate
    # agg_data = aggregate_vols(jsonl_rootdir, banned_extentions)
    # save_jsonl(agg_data, agg_filename)

    # # split
    # train_data, test_data = train_test_split(agg_filename, train_ratio=0.8, seed=42)
    # save_jsonl(train_data, os.path.join(jsonl_rootdir, "train.jsonl"))
    # save_jsonl(test_data, os.path.join(jsonl_rootdir, "test.jsonl"))

    # test_filename = "/datasets/PMC-15M/processed/test_clean_agg.jsonl"
    # val_data, test_data = train_test_split(test_filename, train_ratio = 0.75, seed = 42)
    # save_jsonl(val_data, os.path.join(jsonl_rootdir, "val_clean.jsonl"))
    # save_jsonl(test_data, os.path.join(jsonl_rootdir, "test_clean.jsonl"))

    test_filename = "/datasets/PMC-15M/processed/test_clean.jsonl"
    val_data, test_data = train_test_split(test_filename, train_ratio = 0.5, seed = 42)
    save_jsonl(val_data, os.path.join(jsonl_rootdir, "test_clean_1.jsonl"))
    save_jsonl(test_data, os.path.join(jsonl_rootdir, "test_clean_2.jsonl"))

    # # un-split
    # filename1 = os.path.join(jsonl_rootdir, "val_clean.jsonl")
    # filename2 = os.path.join(jsonl_rootdir, "test_clean.jsonl")
    # outputfilename = os.path.join(jsonl_rootdir, "test_clean_agg.jsonl")
    # aggregate_files(filename1, filename2, outputfilename)

    # # create dummy sets for debugging
    # create_dummy(os.path.join(jsonl_rootdir, "train.jsonl"), 1000)
    # create_dummy(os.path.join(jsonl_rootdir, "test.jsonl"), 500)
