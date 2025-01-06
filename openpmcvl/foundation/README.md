# Foundation Package

This package contains tools to download and extract image-text pairs from PubMedCentral Open-access articles.
Moreover, you can create metadata and train-val-test splits with this package.

The code in this package is a modified version of [Build-PMC-OA](https://github.com/WeixiongLin/Build-PMC-OA).
Please refer to the original repository for more information, and cite their paper[[1]](#cite) if you use this package.

- [Foundation Package](#foundation-package)
  - [Installation](#installation)
  - [Limitation](#limitation)
  - [Cleaning Data](#cleaning-data)
  - [Structure](#structure)
  - [Cite](#cite)


## Installation

1. Setup virtual environment.
After installing the general virtual environment for `openpmcvl` (instructions in the main `README.md`),
install the specific requirements for this package with
```bash
source path/to/venv/bin/activate
pip install -r openpmcvl/foundation/requirements.txt
```

2. Run the script.

```bash
source path/to/venv/bin/activate  # activate venv
cd openpmcvl/foundation
python src/fetch_oa.py --volumes 0 1 2  # Choose volumes of 0,1,2 only
nohup python -u src/fetch_oa.py --extraction-dir path/to/output/dir --volumes 0 > output.txt  # write output to a file
```


## Limitation

1. Some of the paper are only presented in pdf formart, the figures in those would not be obtained by this pipeline
2. We do not provide the capability to download media files other than images, such as suffix mp4, avi; however, the original repository provides this capability.


## Cleaning Data
Following the instructions in [Installation](#installation), you must have image-caption pairs parsed from the articles stored in `jsonl` files.
Moreover, most images listed in the `jsonl` files must be downloaded.
Some images may not have been downloaded due to various reasons one of which is that the image might not actually exist in its expected URL.
Moreover, the `media_name` key in in the `jsonl` entries point to a directory where the image is stored, not the image file itself.

Hence, at this step, the entries need to be cleaned; the non-exitent images should be removed `jsonl` files, and `media_name` should be corrected to point to the image file itself.
Furthermore, we remove captions for the `jsonl` files and, instead, store them in separate text files stored in a directory called `captions`; this significantly reduces GPU memory usage during model training.

You can clean the downloaded data by running below command:
```bash
python src/clean/image_path_url_caption_sep.py  --license-dir path/to/where/jsonl/files/are/stored --volumes 1 2 3 4 5 6 7 8 9 10 11
ln -s path/to/where/jsonl/files/are/stored/figures path/to/where/jsonl/files/are/stored/processed/figures
ln -s path/to/where/jsonl/files/are/stored/captions path/to/where/jsonl/files/are/stored/processed/captions
```
The above command saves the cleaned entries in `jsonl` files in new directory called `processed` under `license-dir`.
Then, it creates symbolic links to folders containing the figures and captions inside the `processed` folder.

After cleaning the data, you can split them into train, validation and test sets by running:
```bash
python src/clean/train_test_split.py  --jsonl-rootdir path/to/where/jsonl/files/are/stored/processed --accepted-exts jpg png
```
A slurm script is provided for both of these commands in `openpmcvl/foundation/src/clean/run.slrm`.

As a final test, you can try loading all images listed in the cleaned splits (i.e. the split's `jsonl` file), and remove the entries whose images don't load for any reason.
To do this, please run below command.
```bash
srun python -u test/test_loadability.py  --root-dir $PMCVL_ROOT_DIR --input-split test_clean --clean-split test_cleaner --mode parallel
```
`PMCVL_ROOT_DIR` is an environment variable that contains the root directory of the dataset (i.e. the path where the splits' `jsonl` files are stored).
This script can run on a single CPU core or on all available CPU cores in parallel. If you set `--mode parallel` it runs the parallel script, and `--mode single` runs on single core.


## Structure

```bash
src/
  |--fetch_oa.py: Main script to download articles and extract <img, caption> pairs.
  |--args/
  | |--args_oa.py: Configures for pipeline
  |--parser/
  | |--parse_oa.py: Parse web pages into list of <img, caption> pairs
  |--utils/
  | |--io.py: Read and write list of <img, caption> pairs to jsonl file.
```


## Cite
```bash
@article{lin2023pmc,
  title={PMC-CLIP: Contrastive Language-Image Pre-training using Biomedical Documents},
  author={Lin, Weixiong and Zhao, Ziheng and Zhang, Xiaoman and Wu, Chaoyi and Zhang, Ya and Wang, Yanfeng and Xie, Weidi},
  journal={arXiv preprint arXiv:2303.07240},
  year={2023}
}
```
