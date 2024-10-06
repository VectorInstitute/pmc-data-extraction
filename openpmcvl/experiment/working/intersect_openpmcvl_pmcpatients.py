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
    return list(set(pmids))

def get_pmcpatients_pmids(root_dir):
    # open PMC_IDs and PAR_PMIDs from metadata
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PMIDs.json")) as file:
        pmids = json.load(file)
    with open(os.path.join(root_dir, "PMC-Patients-MetaData/PAR_PMIDs.json")) as file:
        par_pmids = json.load(file)
    return list(set(pmids)), list(set(par_pmids))


if __name__ == "__main__":
    print("Loading PMC-Patients PMIDs...")
    pmcpatients_pmids, pmcpatients_corpus_pmids = get_pmcpatients_pmids(root_dir="/projects/multimodal/datasets/pmc_patients")
    pmcpatients_pmids = set(pmcpatients_pmids)
    pmcpatients_corpus_pmids = set(pmcpatients_corpus_pmids)
    print(f"Loaded {len(pmcpatients_pmids)} Patient PMIDs and {len(pmcpatients_corpus_pmids)} Corpus PMIDs.")

    print("Loading OpenPMC-VL(train+val splits) PMIDs...")
    openpmcvl_pmids = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="train_clean")
    openpmcvl_pmids_val = get_openpmcvl_article_ids(root_dir="/datasets/PMC-15M/processed/", split="val_clean")
    openpmcvl_pmids.extend(openpmcvl_pmids_val)
    openpmcvl_pmids = set(openpmcvl_pmids)
    print(f"Loaded {len(openpmcvl_pmids)} train+val PMIDs.")

    # count common articles in pmcpatients and openpmcvl
    print("Computing the differece of sets...")
    clean_pmids = pmcpatients_pmids.difference(openpmcvl_pmids)
    print(f"Number of valid PMIDs in PMC-Patients that DON'T exist in OpenPMC-VL train+val splits: {len(clean_pmids)}")
    print(f"Number of common PMIDs in PMC-Patients that DO exist in OpenPMC-VL train+val splits: {len(pmcpatients_pmids) - len(clean_pmids)}")

    clean_corpus_pmids = pmcpatients_corpus_pmids.difference(openpmcvl_pmids)
    print(f"Number of valid PMIDs in PMC-Patients Corpus that DON'T exist in OpenPMC-VL train+val splits: {len(clean_corpus_pmids)}")
    print(f"Number of common PMIDs in PMC-Patients Corpus that DO exist in OpenPMC-VL train+val splits: {len(pmcpatients_corpus_pmids) - len(clean_corpus_pmids)}")

    # save pmids after exclusion
    print("Saving PMIDs after exclusion...")
    with open("/datasets/PMC-15M/PMCPatients_PMIDs_set.json", "w") as outfile:
        json.dump(list(clean_pmids), outfile)
    print("Saved data in /datasets/PMC-15M/PMCPatients_PMIDs_set.json")
    with open("/datasets/PMC-15M/PMCPatients_PAR_PMIDs_set.json", "w") as outfile:
        json.dump(list(clean_corpus_pmids), outfile)
    print("Saved data in /datasets/PMC-15M/PMCPatients_PAR_PMIDs_set.json")
