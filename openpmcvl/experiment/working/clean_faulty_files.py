"""Open all files listed in OpenPMC-VL and delete faulty ones."""
import os
import json
from PIL import Image
from tqdm import tqdm
import multiprocess as mp


Image.MAX_IMAGE_PIXELS = None


def load_split(root_dir, split):
    """Load entries of a given split."""
    data_path = os.path.join(root_dir, f"{split}.jsonl")
    with open(data_path, encoding="utf-8") as file:
        entries = [json.loads(line) for line in file.readlines()]
    return entries


def save_jsonl(data, filename):
    """Save given data in jsonl format."""
    with open(filename, "w") as outfile:
        for sample in data:
            json.dump(sample, outfile)
            outfile.write("\n")


def remove_faulty_files(entries):
    """Open all files in the given split and remove faulty ones."""
    clean_entries = []
    # load image and captions
    for entry in tqdm(entries, total=len(entries), desc=f"cleaning {input_split} split"):
        try:
            img_path = os.path.join(root_dir, "figures", entry["media_name"])
            cap_path = os.path.join(root_dir, "captions", entry["caption_name"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")
            with open(cap_path, encoding="utf-8") as file:
                caption = file.read()
            clean_entries.append(entry)
        except Exception as e:
            print(
                f"Error loading image or caption: image_path={img_path} caption_path={cap_path}",
                "\n",
                e,
                "\nRemoving entry from entrylist...",
            )
    print(f"{len(entries) - len(clean_entries)} entries were removed.")
    return clean_entries


def main(input_split, clean_split):
    """Remove faulty files."""
    global root_dir

    # load split
    entries = load_split(root_dir, input_split)

    # remove faulty files
    clean_entries = remove_faulty_files(entries)

    # save clean entrylist
    print("Saving clean entrylist...")
    filename = os.path.join(root_dir, f"{clean_split}.jsonl")
    save_jsonl(clean_entries, filename)
    print(f"Saved clean entrylist in {filename}")


def main_parallel(input_split, clean_split, nprocess):
    """Remove faulty files."""
    global root_dir

    # load split
    print(f"Loading {input_split} split...")
    entries = load_split(root_dir, input_split)

    # slice entries to the number of tasks
    print("Distributing entries...")
    sublength = (len(entries) + nprocess) // nprocess
    args = []
    for idx in range(0, len(entries), sublength):
        args.append(entries[idx:(idx+sublength)])

    # run jobs in parallel
    with mp.Pool(processes=nprocess) as pool:
        results = pool.map(remove_faulty_files, args)  # list x list x dictionary == nprocess x entries per process x entry

    # aggregate results
    clean_entries = []
    for proc in results:
        clean_entries.extend(proc)

    # save clean entrylist
    print("Saving clean entrylist...")
    filename = os.path.join(root_dir, f"{clean_split}.jsonl")
    save_jsonl(clean_entries, filename)
    print(f"Saved clean entrylist in {filename}")


if __name__ == "__main__":
    root_dir = os.getenv("PMCVL_ROOT_DIR")
    assert root_dir is not None, f"Please enter root directory of OpenPMC-VL dataset in `PMCVL_ROOT_DIR` environment variable."
    input_split = "test_clean"
    clean_split = "test_cleaner"

    # # single core
    # main(input_split, clean_split)

    # multi core
    nprocess = os.environ.get("SLURM_CPUS_PER_TASK")
    if nprocess is None:
        print("Please set the number of CPUs in environment variable `SLURM_CPUS_PER_TASK`.")
        exit(0)
    main_parallel(input_split, clean_split, nprocess=int(nprocess))