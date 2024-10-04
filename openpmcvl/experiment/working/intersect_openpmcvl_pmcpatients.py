"""Find out how many of the articles in PMC-Patients dataset also exist in OpenPMC-VL test split."""
import os
import json
from tqdm import tqdm


def get_openpmcvl_article_ids(root_dir: str, split: str):
    """Get a list of PMIDs of articles in OpenPMC-VL dataset."""
    # load split
    with open(os.path.join(root_dir, f"{split}.jsonl")) as file:
        data = [json.loads(line) for line in tqdm(file)]
    pmids = [sample["PMC_ID"].replace("PMC", "") for sample in data]
    return pmids

def get_pmcpatients_pmids(root_dir):
    # open PMC_IDs and PAR_PMIDs from metadata
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PMIDs.json")) as file:
        pmids = json.load(file)
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PAR_PMIDs.json")) as file:
        par_pmids = json.load(file)
    return pmids, par_pmids

if __name__ == "__main__":
    print("Loading PMC-Patients PMIDs...")
    pmcpatients_pmids, pmcpatients_corpus_pmids = get_pmcpatients_pmids(root_dir="/projects/multimodal/datasets/pmc_patients")

    print("Loading OpenPMC-VL(train+val splits) PMIDs...")
    openpmcvl_pmids = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="train_clean")
    openpmcvl_pmids_val = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="val_clean")
    openpmcvl_pmids.extend(openpmcvl_pmids_val)

    # count common articles in pmcpatients and openpmcvl
    print("Counting common articles in both sets...")
    clean_pmids = pmcpatients_pmids.copy()
    clean_corpus_pmids = pmcpatients_corpus_pmids.copy()
    counter_patients = 0
    counter_corpus = 0
    for id in tqdm(openpmcvl_pmids):
        if id in pmcpatients_pmids:
            counter_patients += 1
            clean_pmids.remove(id)
        if id in pmcpatients_corpus_pmids:
            counter_corpus += 1
            clean_corpus_pmids.remove(id)
    print(f"Number of common PMIDs in PMC-Patients and OpenPMC-VL train+val splits: {counter_patients}")
    print(f"Number of common PMIDs in PMC-Patients Corpus and OpenPMC-VL train+val splits: {counter_corpus}")
    print(f"Number of remaining articles in PMC-Patients after removing the intersection with OpenPMC-VL train+val splits: {len(pmcpatients_pmids) - counter_patients}")
    print(f"Number of remaining articles in PMC-Patients Corpus after removing the intersection with OpenPMC-VL train+val splits: {len(pmcpatients_corpus_pmids) - counter_corpus}")

    # save pmids after exclusion
    print("Saving PMIDs after exclusion...")
    with open("/datasets/PMC-15M/PMCPatients_PMIDs.json", "w") as outfile:
        json.dump(clean_pmids, outfile)
    print("Saved data in /datasets/PMC-15M/PMCPatients_PMIDs.json")
    with open("/datasets/PMC-15M/PMCPatients_PAR_PMIDs.json", "w") as outfile:
        json.dump(clean_corpus_pmids, outfile)
    print("Saved data in /datasets/PMC-15M/PMCPatients_PAR_PMIDs.json")
