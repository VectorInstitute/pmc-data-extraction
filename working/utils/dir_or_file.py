import json
import os

img_rootdir = "/datasets/PMC-15M/figures"

for vol in range(2, 12):
    print(f"VOLUME {vol}")
    json_filename = f"/datasets/PMC-15M/{vol}.jsonl"

    with open(json_filename, "r") as f:
        data = [json.loads(line) for line in f]

    print("ntotal:", len(data))
    count_files = 0
    ninfiles = 0
    for sample in data:
        filename = os.path.join(img_rootdir, sample["media_name"])
        if os.path.isfile(filename):
            count_files += 1
        inname = sample["media_url"].split("/")[-1]
        if not os.path.isfile(os.path.join(filename, inname)):
            ninfiles += 1

    print("nfiles:", count_files)
    print("all in-file?:", ninfiles == 0)
