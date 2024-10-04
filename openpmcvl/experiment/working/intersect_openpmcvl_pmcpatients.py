"""Find out how many of the articles in PMC-Patients dataset also exist in OpenPMC-VL test split."""
import os
import json
from tqdm import tqdm


def get_openpmcvl_article_ids(root_dir: str, split: str):
    """Get a list of PMIDs of articles in OpenPMC-VL dataset."""
    # load split
    with open(os.path.join(root_dir, f"{split}.jsonl")) as file:
        data = [json.loads(line) for line in file]
    pmids = [sample["PMC_ID"].replace("PMC", "") for sample in data]
    return pmids

def get_pmcpatients_pmids(root_dir):
    # open PMC_IDs and PAR_PMIDs from metadata
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PMIDs.json")) as file:
        pmids = json.load(file)
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PAR_PMIDs.json")) as file:
        par_pmids = json.load(file)
    par_pmids.extend(pmids)
    return list(set(par_pmids))

if __name__ == "__main__":
    print("Loading PMC-Patients PMIDs...")
    pmcpatients_pmids = get_pmcpatients_pmids(root_dir="/projects/multimodal/datasets/pmc_patients")

    print("Loading OpenPMC-VL(train+val splits) PMIDs...")
    openpmcvl_pmids = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="train_clean")
    openpmcvl_pmids_val = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="val_clean")
    openpmcvl_pmids.extend(openpmcvl_pmids_val)

    # count common articles in pmcpatients and openpmcvl
    print("Counting common articles in both sets...")
    counter = 0
    for id in tqdm(pmcpatients_pmids):
        if id in openpmcvl_pmids:
            counter += 1
    print(f"Number of common PMIDs in PMC-Patients and OpenPMC-VL train+val splits: {counter}")
    print(f"Number of remaining articles in PMC-Patients after removing the intersection with OpenPMC-VL train+val splits: {len(pmcpatients_pmids) - counter}")