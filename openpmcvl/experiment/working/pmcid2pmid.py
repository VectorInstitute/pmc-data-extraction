"""Convert PMC-IDs of OpenPMC-VL articles to PMIDs."""
import os
import json
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import multiprocess as mp


def load_pmcids(root_dir, split):
    """Load the set of PMCIDs in a split of OpenPMC-VL."""
    filename = os.path.join(root_dir, f"{split}.jsonl")
    print(f"Loading PMCIDs from {filename}...")
    with open(filename, "r") as file:
        pmcids = [json.loads(line)["PMC_ID"] for line in tqdm(file)]
    pmcids = set(pmcids)
    print(f"{len(pmcids)} PMCIDs loaded.")
    return pmcids


def map_pmcid2pmid(pmcids):
    """Convert PMCIDs of OpenPMC-VL to PMIDs."""
    # server url
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    # create a session
    session = requests.Session()
    # define a retry strategy
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Waits 1 second between retries, then 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
    )
    # mount the retry strategy to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # convert pmcid to pmid
    print("Converting PMCIDs to PMID...")
    pmcid2pmid = {}
    pmids = []
    for pmcid in tqdm(pmcids, desc=f"PID#{os.getpid()}"):
        try:
            response = session.get(f"{service_root}?ids={pmcid}&idtype=pmcid&versions=no&format=json")
            if response.status_code == 200:
                json_obj = json.loads(response.text)
                pmid = json_obj["records"][0]["pmid"]
                pmcid2pmid[pmcid] = pmid
                pmids.append(pmid)
        except requests.exceptions.ConnectionError as e:
            print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}")
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


def main():
    # get all pmcids
    pmcids = []
    for split in ["val_clean", "train_clean"]:
        # load pmcids
        pmcids.extend(list(load_pmcids(root_dir="/datasets/PMC-15M/processed/", split=split)))
    # convert pmcid to pmid
    openpmcvl_pmcid2pmid, openpmcvl_pmids = map_pmcid2pmid(list(pmcids))
    # save on disk
    save_json(openpmcvl_pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_clean.json")
    save_json(openpmcvl_pmids, f"/datasets/PMC-15M/processed/pmids_train_val_clean.json")


def get_num_articles(root_dir, split):
    """Get number of articles in a given split of OpenPMC-VL."""
    filename = os.path.join(root_dir, f"{split}.jsonl")
    with open(filename, "r") as file:
        pmcids = [json.loads(line)["PMC_ID"] for line in tqdm(file)]
    pmcids = set(pmcids)
    return len(pmcids)


def main_parallel(nprocess):
    """Convert PMCID 2 PMID in a distributed manner."""
    # get all pmcids
    pmcids = []
    for split in ["val_clean", "train_clean"]:
        # load pmcids
        pmcids.extend(list(load_pmcids(root_dir="/datasets/PMC-15M/processed/", split=split)))

    # slice pmcids to the number of tasks
    sublength = (len(pmcids) + nprocess) // nprocess
    args = []
    for idx in range(0, len(pmcids), sublength):
        args.append(pmcids[idx:(idx+sublength)])

    # run jobs in parallel
    with mp.Pool(processes=nprocess) as pool:
        results = pool.map(map_pmcid2pmid, args)  # list x tuple x list == nprocess x num_outputs x output_length

    # aggregate results
    pmcid2pmid = {}
    pmids = []
    for proc in results:
        pmcid2pmid.update(proc[0])
        pmids.extend(proc[1])

    # save results
    save_json(pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_clean.json")
    save_json(pmids, f"/datasets/PMC-15M/processed/pmids_train_val_clean.json")


def test_parallel():
    """Compare results of single-process and parallel runs."""
    pmc2pmid_single = load_json("/datasets/PMC-15M/processed/pmc2pmid_train_val.json")
    pmid_single = load_json("/datasets/PMC-15M/processed/pmids_train_val.json")
    pmc2pmid_parallel = load_json("/datasets/PMC-15M/processed/pmc2pmid_train_val_parallel.json")
    pmid_parallel = load_json("/datasets/PMC-15M/processed/pmids_train_val_parallel.json")

    assert pmid_single.sort() == pmid_parallel.sort()
    assert pmc2pmid_single.keys() == pmc2pmid_parallel.keys()


if __name__ == "__main__":
    # single process
    # main()

    # multi-process
    nprocess = os.environ.get("SLURM_CPUS_PER_TASK")
    if nprocess is None:
        print("Please set the number of CPUs in environment variable `SLURM_CPUS_PER_TASK`.")
        exit(0)
    main_parallel(nprocess=int(nprocess))

    # test multi-process
    # test_parallel()
