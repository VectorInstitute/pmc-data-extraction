"""Find out how many of the articles in PMC-Patients dataset also exist in OpenPMC-VL test split."""
import os
import json
from tqdm import tqdm


def get_openpmcvl_pmc_ids(root_dir: str, split: str):
    """Get a list of PMIDs of articles in OpenPMC-VL dataset."""
    with open(os.path.join(root_dir, f"{split}.jsonl")) as file:
        pmids = [json.loads(line)["PMC_ID"] for line in tqdm(file)]
    return set(pmids)


def get_pmcpatients_pmids(root_dir):
    """Load patient PMIDs and PAR_PMIDs from metadata"""
    pmids = load_json(os.path.join(root_dir, "PMC-Patients-MetaData/PMIDs.json"))
    par_pmids = load_json(os.path.join(root_dir, "PMC-Patients-MetaData/PAR_PMIDs.json"))
    return set(pmids), set(par_pmids)


def get_openpmcvl_pmids(root_dir, split):
    """Load PMIDs of OpenPMC-VL"""
    return set(load_json(os.path.join(root_dir, f"pmids_{split}.json")))


def load_json(filename):
    """Load json data from file."""
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def save_json(data, filename):
    """Save given data in given file in json format."""
    with open(filename, "w") as outfile:
        json.dump(data, outfile)
    print(f"Saved data in {filename}")


def compute_num_patients(root_dir):
    """Compute number of patients that are extracted from valid articles that don't intersect with OpenPMC-VL train and val splits."""
    pass


def outersect_patients_openpmcvl():
    """Find and save the difference of article sets in PMCPatients and OpenPMC-VL."""
    print("Loading PMC-Patients PMIDs...")
    pmcpatients_pmids, pmcpatients_corpus_pmids = get_pmcpatients_pmids(root_dir="/projects/multimodal/datasets/pmc_patients")
    print(f"Loaded {len(pmcpatients_pmids)} Patient PMIDs and {len(pmcpatients_corpus_pmids)} Corpus PMIDs.")

    print("Loading OpenPMC-VL(train+val splits) PMIDs...")
    openpmcvl_pmids = get_openpmcvl_pmids(root_dir="/datasets/PMC-15M/processed/", split="train_val_clean")
    print(f"Loaded {len(openpmcvl_pmids)} train+val PMIDs.")

    # count different articles in pmcpatients and openpmcvl
    print("Computing the differece of article sets...")
    clean_pmcpatients_pmids = pmcpatients_pmids.difference(openpmcvl_pmids)
    print(f"Number of valid PMIDs in PMC-Patients that DON'T exist in OpenPMC-VL train+val splits: {len(clean_pmcpatients_pmids)}")
    print(f"Number of common PMIDs in PMC-Patients that DO exist in OpenPMC-VL train+val splits: {len(pmcpatients_pmids) - len(clean_pmcpatients_pmids)}")

    clean_pmcpatients_corpus_pmids = pmcpatients_corpus_pmids.difference(openpmcvl_pmids)
    print(f"Number of valid PMIDs in PMC-Patients Corpus that DON'T exist in OpenPMC-VL train+val splits: {len(clean_pmcpatients_corpus_pmids)}")
    print(f"Number of common PMIDs in PMC-Patients Corpus that DO exist in OpenPMC-VL train+val splits: {len(pmcpatients_corpus_pmids) - len(clean_pmcpatients_corpus_pmids)}")

    # save pmids after exclusion
    print("Saving patient PMIDs after exclusion...")
    save_json(list(clean_pmcpatients_pmids), "/datasets/PMC-15M/PMCPatients_PMIDs.json")
    save_json(list(clean_pmcpatients_corpus_pmids), "/datasets/PMC-15M/PMCPatients_PAR_PMIDs.json")


if __name__ == "__main__":
    outersect_patients_openpmcvl()

