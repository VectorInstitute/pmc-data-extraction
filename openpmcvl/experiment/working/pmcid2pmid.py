"""Convert PMC-IDs of OpenPMC-VL articles to PMIDs."""
import os
import json
from tqdm import tqdm
import requests
import pandas as pd


def map_pmcid2pmid(root_dir, split):
    """Convert PMCIDs of OpenPMC-VL to PMIDs."""
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

    # load PMCIDs
    filename = os.path.join(root_dir, f"{split}.jsonl")
    print(f"Loading PMCIDs from {filename}...")
    with open(filename, "r") as file:
        pmcids = [json.loads(line)["PMC_ID"] for line in tqdm(file)]

    # convert pmcid to pmid
    print("Converting PMCIDs to PMID...")
    pmcid2pmid = {}
    pmids = []
    for pmcid in tqdm(pmcids):
        response = requests.get(f"{service_root}?ids={pmcid}&idtype=pmcid&versions=no&format=json")
        if response.status_code == 200:
            json_obj = json.loads(response.text)
            try:
                pmid = json_obj["records"][0]["pmid"]
            except Exception as e:
                print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}")
            pmcid2pmid[pmcid] = pmid
            pmids.append(pmid)
    print(f"{len(pmids)}/{len(pmcids)} PMCIDs converted.")

    return pmcid2pmid, pmids


def save_json(data, filename):
    """Save given data in given file in json format."""
    with open(filename, "w") as outfile:
        json.dump(data, outfile)
    print(f"Saved data in {filename}")


def load_json(filename):
    """Load json data from given filename."""
    with open(filename, "r") as file:
        data = json.loads(file.read())
    return data


if __name__ == "__main__":
    for split in ["val_clean", "train_clean"]:
        openpmcvl_pmcid2pmid, openpmcvl_pmids = map_pmcid2pmid(root_dir="/datasets/PMC-15M/processed/", split=split)
        # save on disk
        save_json(openpmcvl_pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_{split}.json")
        save_json(openpmcvl_pmids, f"/datasets/PMC-15M/processed/pmids_{split}.json")

    # concatenate pmids
    pmids_cat = []
    for split in ["val_clean", "train_clean"]:
        pmids_cat.extend(load_json(f"/datasets/PMC-15M/processed/pmids_{split}.json"))
    save_json(pmids_cat, f"/datasets/PMC-15M/processed/pmids_train_val_clean.json")
