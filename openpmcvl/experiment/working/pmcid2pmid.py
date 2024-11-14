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
import argparse


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


def load_pmcids(root_dir, split):
    """Load a list PMC-IDs in a split of OpenPMC-VL."""
    filename = os.path.join(root_dir, f"{split}.jsonl")
    print(f"Loading PMC-IDs from {filename}...")
    with open(filename, "r", encoding="utf-8") as file:
        pmcids = [json.loads(line)["PMC_ID"] for line in tqdm(file)]
    pmcids = list(set(pmcids))
    print(f"{len(pmcids)} PMC-IDs loaded.")
    return pmcids


def map_pmcid2pmid(pmcids):
    """Convert PMC-IDs of OpenPMC-VL to PMIDs.

    For each PMC-ID, make a request to an API provided by PubMedCentral to
    convert to PMID.
    """
    # server url
    service_root = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
    # create a session - essential step when running parallel tasks
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
    print("Converting PMC-IDs to PMID...")
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
    print(f"{len(pmids)}/{len(pmcids)} PMC-IDs converted.")

    return pmcid2pmid, pmids


def main():
    """Load PMC-IDs of OpenPMC-VL splits and convert to PMIDs.

    Meant to run on a single process.
    Alternate functions are provided for single-node-multi-core and
    multi-node-multi-core runs.
    """
    root_dir = os.getenv("PMCVL_ROOT_DIR", "")

    # get all pmcids
    pmcids = []
    for split in ["train_cleaner", "val_cleaner"]:
        # load pmcids
        pmcids.extend(load_pmcids(root_dir=root_dir, split=split))
    pmcids = sorted(list(set(pmcids)))
    print(f"{len(pmcids)} PMC-IDs loaded.")

    # convert pmcid to pmid
    pmcid2pmid, pmids = map_pmcid2pmid(pmcids)

    # save on disk
    save_json(pmcid2pmid, os.path.join(root_dir, "pmc2pmid_single.json"))
    save_json(pmids, os.path.join(root_dir, "pmids_single.json"))


def main_parallel(nprocess):
    """Load PMC-IDs of OpenPMC-VL splits and convert to PMIDs.

    Meant to run on a single node but multiple cores in parallel.

    Parameters
    ----------
    nprocess: int
        Number of parallel processes on which to distribute the task.
        Maximum amount is equal to the number of available CPUs on the node.
    """
    root_dir = os.getenv("PMCVL_ROOT_DIR", "")

    # get all pmcids
    pmcids = []
    for split in ["train_cleaner", "val_cleaner"]:
        # load pmcids
        pmcids.extend(load_pmcids(root_dir=root_dir, split=split))
    pmcids = sorted(list(set(pmcids)))
    print(f"{len(pmcids)} PMC-IDs loaded.")

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

    # save on disk
    save_json(pmcid2pmid, os.path.join(root_dir, "pmc2pmid_parallel.json"))
    save_json(pmids, os.path.join(root_dir, "pmids_parallel.json"))


def main_multinode():
    """Load PMC-IDs of OpenPMC-VL splits and convert to PMIDs.

    Meant to run on multiple nodes and cores in parallel.
    Stores results per node in separate files. Another function
    (`concat_multinode`) is provided to concatenate the per-node results into
    single files.
    Only run via submitting a job on Slurm since several parameters including
    the number of nodes and CPUs per node are obtained from Slurm environment
    variables.
    """
    root_dir = os.getenv("PMCVL_ROOT_DIR", "")

    # get all pmcids
    pmcids = []
    for split in ["train_cleaner", "val_cleaner"]:
        # load pmcids
        pmcids.extend(load_pmcids(root_dir=root_dir, split=split))
    pmcids = sorted(list(set(pmcids)))
    print(f"{len(pmcids)} PMC-IDs loaded.")

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

    # save results per node
    save_json(pmcid2pmid, os.path.join(root_dir, f"pmc2pmid_multinode_{node_id}.json"))
    save_json(pmids, os.path.join(root_dir, f"pmids_multinode_{node_id}.json"))


def concat_multinode(num_nodes=None):
    """Concatenate results of multiple nodes.

    Only run after all nodes' results are saved on disk.
    """
    root_dir = os.getenv("PMCVL_ROOT_DIR", "")
    if num_nodes is None:
        num_nodes = int(os.environ.get("SLURM_NNODES"))

    pmcid2pmid = {}
    pmids = []
    for node_id in range(num_nodes):
        pmcid2pmid.update(load_json(os.path.join(root_dir, f"pmc2pmid_multinode_{node_id}.json")))
        pmids.extend(load_json(os.path.join(root_dir, f"pmids_multinode_{node_id}.json")))

    # save results of all nodes
    save_json(pmcid2pmid, os.path.join(root_dir, "pmc2pmid_multinode.json"))
    save_json(pmids, os.path.join(root_dir, "pmids_multinode.json"))

    # delete per-node files
    for node_id in range(num_nodes):
        os.remove(os.path.join(root_dir, f"pmc2pmid_multinode_{node_id}.json"))
        os.remove(os.path.join(root_dir, f"pmids_multinode_{node_id}.json"))


def test_modes():
    """Compare results of the three modes.

    Three main functions are provided to convert OpenPMC-VL PMC-IDs to PMIDs in:
        1. a single process
        2. parallel processes on a single node
        3. parallel processes on multiple nodes
    This fucntion tests if all modes end up with the same PMIDs and PMC-ID-to-PMID maps.
    """
    root_dir = os.getenv("PMCVL_ROOT_DIR", "")

    # load results of all three modes from disk
    pmc2pmid_single = load_json(os.path.join(root_dir, "pmc2pmid_single.json"))
    pmid_single = load_json(os.path.join(root_dir, "pmids_single.json"))
    pmc2pmid_parallel = load_json(os.path.join(root_dir, "pmc2pmid_parallel.json"))
    pmid_parallel = load_json(os.path.join(root_dir, "pmids_parallel.json"))
    pmc2pmid_multinode = load_json(os.path.join(root_dir, "pmc2pmid_multinode.json"))
    pmid_multinode = load_json(os.path.join(root_dir, "pmids_multinode.json"))

    print(f"Number of pmids in single: {len(pmid_single)}")
    print(f"Number of pmids in parallel: {len(pmid_parallel)}")
    print(f"Number of pmids in multinode: {len(pmid_multinode)}")

    # find repeated values in pmids
    repeated_single = [(item, count) for item, count in collections.Counter(pmid_single).items() if count > 1]
    repeated_parallel = [(item, count) for item, count in collections.Counter(pmid_parallel).items() if count > 1]
    repeated_multinode = [(item, count) for item, count in collections.Counter(pmid_multinode).items() if count > 1]
    print(f"Repeated pmids in single: {repeated_single}")
    print(f"Repeated pmids in parallel: {repeated_parallel}")
    print(f"Repeated pmids in multinode: {repeated_multinode}")

    assert pmid_single == pmid_parallel
    assert pmid_single == pmid_multinode
    assert all((pmc2pmid_single.get(k) == v for k, v in pmc2pmid_parallel.items()))
    assert all((pmc2pmid_single.get(k) == v for k, v in pmc2pmid_multinode.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "parallel", "multinode", "concat_multinode", "test"])
    cli_args = parser.parse_args()

    if cli_args.mode == "single":
        # single process
        main()
    elif cli_args.mode == "parallel":
        # single-node multi-core
        nprocess = os.environ.get("SLURM_CPUS_PER_TASK")
        if nprocess is None:
            print("Please set the number of CPUs in environment variable `SLURM_CPUS_PER_TASK`.")
            exit(0)
        main_parallel(nprocess=int(nprocess))
    elif cli_args.mode == "multinode":
        # multi-node multi-core
        main_multinode()  # each node's result is saved separately
    elif cli_args.mode == "concat_multinode":
        # concatenate per-node results into single files
        concat_multinode(num_nodes=None)
    elif cli_args.mode == "test":
        # compare the results of all three modes
        test_modes()
