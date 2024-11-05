"""Find out how many of the articles in PMC-Patients dataset also exist in OpenPMC-VL test split."""
import os
import json
from tqdm import tqdm
import pandas as pd


def get_openpmcvl_pmc_ids(root_dir: str, split: str):
    """Get a list of PMC_IDs of articles in OpenPMC-VL dataset."""
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


def load_jsonl(filename):
    """Load jsonl data from file."""
    with open(filename, "r") as file:
        data = [json.loads(line) for line in file]
    return data


def save_json(data, filename):
    """Save given data in given file in json format."""
    with open(filename, "w") as outfile:
        json.dump(data, outfile)
    print(f"Saved data in {filename}")


def save_jsonl(data, filename):
    """Save given data in given file in jsonl format."""
    with open(filename, "w") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")
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


def clean_queries():
    """Clean queries from overlapping patients."""
    # load openpmcvl pmids
    pmid_openpmcvl = load_json(os.path.join(os.getenv("PMCVL_ROOT_DIR", ""), "pmids_multinode.json"))

    # load pmc-patients queries
    patients_rootdir = os.getenv("PMCPATIENTS_ROOT_DIR", "")
    patients_queries = load_jsonl(os.path.join(patients_rootdir, "queries", "test_queries.jsonl"))
    patients_queries = pd.DataFrame.from_records(patients_queries)

    # extract pmids
    patients_queries["pmid"] = patients_queries["_id"].apply(lambda uid: uid.split("-")[0])
    pmid_queries = patients_queries["pmid"].tolist()
    # find the intersection of both pmid lists
    pmid_common = list(set(pmid_openpmcvl).intersection(set(pmid_queries)))
    # remove common pmids from queries
    patients_queries = patients_queries.loc[patients_queries["pmid"].apply(lambda pmid: pmid not in pmid_common)]
    patients_queries = patients_queries.drop(columns=["pmid"])

    # save clean queries
    outfile = os.path.join(f"{patients_rootdir}-Clean", "queries", "test_queries.jsonl")
    save_jsonl(patients_queries.to_dict(orient="records"), outfile)


def clean_qrels():
    """Clean qrels from overlapping patients."""
    # load openpmcvl pmids
    pmid_openpmcvl = load_json(os.path.join(os.getenv("PMCVL_ROOT_DIR", ""), "pmids_multinode.json"))

    # load pmc-patients queries
    patients_rootdir = os.getenv("PMCPATIENTS_ROOT_DIR", "")
    patients_qrels = pd.read_csv(os.path.join(patients_rootdir, "PPR", "qrels_test.tsv"), sep="\t")
    print(patients_qrels)

    # extract query and corpus pmids
    patients_qrels["query-pmid"] = patients_qrels["query-id"].apply(lambda uid: uid.split("-")[0])
    patients_qrels["corpus-pmid"] = patients_qrels["corpus-id"].apply(lambda uid: uid.split("-")[0])
    pmid_query = patients_qrels["query-pmid"].tolist()
    pmid_corpus = patients_qrels["corpus-pmid"].tolist()
    # find the intersection of both pmid lists
    pmid_common_query = list(set(pmid_openpmcvl).intersection(set(pmid_query)))
    pmid_common_corpus = list(set(pmid_openpmcvl).intersection(set(pmid_corpus)))
    # remove common pmids from qrels
    patients_qrels = patients_qrels.loc[patients_qrels["query-pmid"].apply(lambda pmid: pmid not in pmid_common_query)]
    patients_qrels = patients_qrels.loc[patients_qrels["corpus-pmid"].apply(lambda pmid: pmid not in pmid_common_corpus)]
    patients_qrels = patients_qrels.drop(columns=["query-pmid", "corpus-pmid"])
    print(patients_qrels)

    # save clean qrels
    patients_qrels.to_csv(os.path.join(f"{patients_rootdir}-Clean", "PPR", "qrels_test.tsv"), sep="\t")


def clean_corpus():
    """Clean corpus from overlapping patients."""
    # load openpmcvl pmids
    pmid_openpmcvl = load_json(os.path.join(os.getenv("PMCVL_ROOT_DIR", ""), "pmids_multinode.json"))

    # load pmc-patients queries
    patients_rootdir = os.getenv("PMCPATIENTS_ROOT_DIR", "")
    patients_queries = load_jsonl(os.path.join(patients_rootdir, "PPR", "corpus.jsonl"))
    patients_queries = pd.DataFrame.from_records(patients_queries)

    # extract pmids
    patients_queries["pmid"] = patients_queries["_id"].apply(lambda uid: uid.split("-")[0])
    pmid_queries = patients_queries["pmid"].tolist()
    # find the intersection of both pmid lists
    pmid_common = list(set(pmid_openpmcvl).intersection(set(pmid_queries)))
    # remove common pmids from queries
    patients_queries = patients_queries.loc[patients_queries["pmid"].apply(lambda pmid: pmid not in pmid_common)]
    patients_queries = patients_queries.drop(columns=["pmid"])

    # save clean queries
    outfile = os.path.join(f"{patients_rootdir}-Clean", "PPR", "corpus.jsonl")
    save_jsonl(patients_queries.to_dict(orient="records"), outfile)



if __name__ == "__main__":
    # outersect_patients_openpmcvl()

    # clean_queries()
    # clean_qrels()
    clean_corpus()

