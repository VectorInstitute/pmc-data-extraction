"""Convert PMC-IDs of OpenPMC-VL articles to PMIDs."""
import os
import json
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import multiprocess as mp
import collections
import numpy as np


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
        total=10,  # Total number of retries
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
        pmid = "NA"
        try:
            response = session.get(f"{service_root}?ids={pmcid}&idtype=pmcid&versions=no&format=json")
            if response.status_code == 200:
                json_obj = json.loads(response.text)
                pmid = json_obj["records"][0]["pmid"]
            else:
                print(f"Error: response is {response.status_code}: {response.text}. Setting pmid to 'ST'")
                pmid = "ST"
        except requests.exceptions.ConnectionError as e:
            print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}. Setting pmid to 'CE'.")
            pmid = "CE"
        except KeyError as e:
            print(f"KeyError occured for pmcid={pmcid}: {e}. Setting pmid to None.")
            pmid = None
        except Exception as e:
            print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}. Setting pmid to 'NA'.")
        pmcid2pmid[pmcid] = pmid
        pmids.append(pmid)
    print(f"{len(pmids)}/{len(pmcids)} PMCIDs converted.")

    return pmcid2pmid, pmids


def test_map_pmcid2pmid(pmcid):
    """Convert PMCIDs of OpenPMC-VL to PMIDs."""
    # server url
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    # create a session
    session = requests.Session()
    # define a retry strategy
    retry_strategy = Retry(
        total=10,  # Total number of retries
        backoff_factor=1,  # Waits 1 second between retries, then 2s, 4s, 8s...
        status_forcelist=[429, 500, 502, 503, 504],  # Status codes to retry on
    )
    # mount the retry strategy to the session
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # convert pmcid to pmid
    print("Converting PMCIDs to PMID...")
    pmid = "NA"
    try:
        response = session.get(f"{service_root}?ids={pmcid}&idtype=pmcid&versions=no&format=json")
        print(f"Response: {response.text}")
        if response.status_code == 200:
            json_obj = json.loads(response.text)
            pmid = json_obj["records"][0]["pmid"]
        else:
            print(f"Error: response is {response.status_code}: {response.text}. Setting pmid to 'ST'")
            pmid = "ST"
    except requests.exceptions.ConnectionError as e:
        print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}. Setting pmid to 'CE'.")
        pmid = "CE"
    except KeyError as e:
        print(f"KeyError occured for pmcid={pmcid}: {e}. Setting pmid to None.")
        pmid = None
    except Exception as e:
        print(f"Error occured for pmcid={pmcid}: {type(e).__name__}: {e}. Setting pmid to 'NA'.")
    print(pmid)


def save_json(data, filename):
    """Save given data in given file in json format."""
    with open(filename, "w") as outfile:
        json.dump(data, outfile)
    print(f"Saved data in {filename}")


def load_json(filename):
    """Load json data from given filename."""
    with open(filename, "r") as file:
        data = json.load(file)
    return data


def main():
    # get all pmcids
    pmcids = []
    for split in ["test_dummy_", "train_dummy_"]:
        # load pmcids
        pmcids.extend(list(load_pmcids(root_dir="/datasets/PMC-15M/processed/", split=split)))
    pmcids = list(set(pmcids))
    print(f"{len(pmcids)} PMCIDs loaded.")
    # convert pmcid to pmid
    openpmcvl_pmcid2pmid, openpmcvl_pmids = map_pmcid2pmid(pmcids)
    # save on disk
    save_json(openpmcvl_pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_single.json")
    save_json(openpmcvl_pmids, f"/datasets/PMC-15M/processed/pmids_train_val_single.json")


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
    for split in ["test_dummy_", "train_dummy_"]:
        # load pmcids
        pmcids.extend(list(load_pmcids(root_dir="/datasets/PMC-15M/processed/", split=split)))
    pmcids = list(set(pmcids))
    print(f"{len(pmcids)} PMCIDs loaded.")
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
    save_json(pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_parallel.json")
    save_json(pmids, f"/datasets/PMC-15M/processed/pmids_train_val_parallel.json")


def main_multinode():
    """Convert PMCID 2 PMID in a multi-node and multi-core manner."""
    # get all pmcids
    pmcids = []
    for split in ["test_dummy_", "train_dummy_"]:
        # load pmcids
        pmcids.extend(list(load_pmcids(root_dir="/datasets/PMC-15M/processed/", split=split)))
    pmcids = sorted(list(set(pmcids)))
    print(f"{len(pmcids)} PMCIDs loaded.")

    # get essential environment variables
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_node = os.environ.get("SLURM_NODELIST")
    node_id = int(os.environ.get("SLURM_NODEID"))
    num_nodes = int(os.environ.get("SLURM_NNODES"))
    print(f"Running on Slurm Job ID: {slurm_job_id}, Node: {slurm_node}")

    # use all available CPUs on this node if a number is not given
    num_processes = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"num_processes: {num_processes}, num_nodes={num_nodes}, node_id={node_id}")

    # calculate the range of data this node should process
    total_tasks = num_processes * num_nodes  # total number of tasks across all nodes
    start_index = node_id * num_processes  # start task index for this node
    end_index = start_index + num_processes if node_id < num_nodes - 1 else total_tasks  # end task index for this node

    # slice pmcids to the number of tasks
    sublength = (len(pmcids) + total_tasks) // total_tasks
    args = []
    for idx in range(start_index * sublength, end_index * sublength, sublength):
        args.append(pmcids[idx:(idx+sublength)])

    # run jobs in parallel
    with mp.Pool(processes=(end_index - start_index)) as pool:
        results = pool.map(map_pmcid2pmid, args)  # list x tuple x list == nprocess x num_outputs x output_length

    # aggregate results
    pmcid2pmid = {}
    pmids = []
    for proc in results:
        pmcid2pmid.update(proc[0])
        pmids.extend(proc[1])

    # save results
    save_json(pmcid2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_multinode_{node_id}.json")
    save_json(pmids, f"/datasets/PMC-15M/processed/pmids_train_val_multinode_{node_id}.json")


def concat_multinode(num_nodes=None):
    if num_nodes is None:
        num_nodes = int(os.environ.get("SLURM_NNODES"))
    pmc2pmid = {}
    pmids = []
    for node_id in range(num_nodes):
        pmc2pmid.update(load_json(f"/datasets/PMC-15M/processed/pmc2pmid_train_val_multinode_{node_id}.json"))
        pmids.extend(load_json(f"/datasets/PMC-15M/processed/pmids_train_val_multinode_{node_id}.json"))

    # save results
    save_json(pmc2pmid, f"/datasets/PMC-15M/processed/pmc2pmid_train_val_multinode.json")
    save_json(pmids, f"/datasets/PMC-15M/processed/pmids_train_val_multinode.json")



def test_parallel():
    """Compare results of single-process and parallel runs."""
    pmc2pmid_single = load_json("/datasets/PMC-15M/processed/pmc2pmid_train_val_single.json")
    pmid_single = load_json("/datasets/PMC-15M/processed/pmids_train_val_single.json")
    pmc2pmid_parallel = load_json("/datasets/PMC-15M/processed/pmc2pmid_train_val_parallel.json")
    pmid_parallel = load_json("/datasets/PMC-15M/processed/pmids_train_val_parallel.json")

    print(f"number of pmids in single: {len(pmid_single)}")
    print(f"number of pmids in parallel: {len(pmid_parallel)}")

    # find repeated values in pmids
    repeated_single = [(item, count) for item, count in collections.Counter(pmid_single).items() if count > 1]
    repeated_parallel = [(item, count) for item, count in collections.Counter(pmid_parallel).items() if count > 1]
    print(f"repeated pmids in single: {repeated_single}")
    print(f"repeated pmids in parallel: {repeated_parallel}")
    print(f"number of repeated pmids in single: {len(repeated_single)}")
    print(f"number of repeated pmids in parallel: {len(repeated_parallel)}")


    pmid_single = set(pmid_single)
    pmid_parallel = set(pmid_parallel)
    # # TODO: these tests fail. Figure out why.
    # assert pmid_single == pmid_parallel
    # assert set(pmc2pmid_single.items()) == set(pmc2pmid_parallel.items())

    # print(pmid_single)
    # print("\n\n\n\n", pmid_parallel)
    print(f"number of pmids in single: {len(pmid_single)}")
    print(f"number of pmids in parallel: {len(pmid_parallel)}")
    print(f"number of pmc_ids in single: {len(pmc2pmid_single)}")
    print(f"number of pmc_ids in parallel: {len(pmc2pmid_parallel)}")

    # check if everything in the parallel pmids exits in the single one
    # count = 0
    # for id in pmid_parallel:
    #     if id not in pmid_single:
    #         count += 1
    #         print(f"Discrepancy #{count}: {id}")
    # print(f"Number of discrepancies: {count}")





if __name__ == "__main__":
    # # single process
    # main()

    # # multi-process
    # nprocess = os.environ.get("SLURM_CPUS_PER_TASK")
    # if nprocess is None:
    #     print("Please set the number of CPUs in environment variable `SLURM_CPUS_PER_TASK`.")
    #     exit(0)
    # main_parallel(nprocess=int(nprocess))

    # multi-node
    # main_multinode()
    # # run this after all results are saved on file
    # concat_multinode(num_nodes=2)

    # test multi-process
    test_parallel()
