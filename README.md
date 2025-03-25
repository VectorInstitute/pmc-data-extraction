# Open-PMC

----------------------------------------------------------------------------------------

[![code checks](https://github.com/VectorInstitute/aieng-template/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/pmc-data-extraction/actions/workflows/code_checks.yml)
[![integration tests](https://github.com/VectorInstitute/aieng-template/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/pmc-data-extraction/actions/workflows/integration_tests.yml)
[![license](https://img.shields.io/github/license/VectorInstitute/aieng-template.svg)](https://github.com/VectorInstitute/pmc-data-extraction/blob/main/LICENSE.md)

<div align="center">
    <img src="https://github.com/VectorInstitute/pmc-data-extraction/blob/0a969136344a07267bb558d01f3fe76b36b93e1a/media/open-pmc-pipeline.png?raw=true" 
     alt="Open-PMC Pipeline" 
     width="1000" />
</div>

A toolkit to download, augment, and benchmark Open-PMC; a large dataset of image-text pairs extracted from open-access scientific articles on PubMedCentral.

For more details, see the following resources:
- **arXiv Paper:** [http://arxiv.org/abs/2503.14377](http://arxiv.org/abs/2503.14377)
- **Dataset:** [https://huggingface.co/datasets/vector-institute/open-pmc](https://huggingface.co/datasets/vector-institute/open-pmc)
- **Model Checkpoint:** [https://huggingface.co/vector-institute/open-pmc-clip](https://huggingface.co/vector-institute/open-pmc-clip)

## Table of Contents

1. [Installing Dependencies](#installing-dependencies)
2. [Download and Parse Image-Caption Pairs](#download-and-parse-image-caption-pairs-from-pubmed-articles)
3. [Run Benchmarking Experiments](#run-benchmarking-experiments)
4. [Citation](#citation)

## Installing dependencies

We use
[poetry](https://python-poetry.org/docs/#installation)
for dependency management. Please make sure it is installed.
Then, follow below instructions to set up your virtual environment.

1. Create a venv with python3.10 and activate it.
```bash
python --version  # must print 3.10
python -m venv <your-venv-name>
source <your-venv-name>/bin/activate
```

2. Navigate to the root directory of pmc-data-extraction repository and install dependencies.
Two of the required dependencies are [mmlearn](https://github.com/VectorInstitute/mmlearn) and [open_clip](https://github.com/mlfoundations/open_clip).
You have the option to either install them with `pip` or from source.

To install `mmlearn` and `open_clip` with `pip`, run
```bash
cd path/to/pmc-data-extraction
pip install --upgrade pip
poetry install --no-root --with test,open_clip,mmlearn --all-extras
```
then skip to step 6: Check Installations.

To install `mmlearn` and `open_clip` from source, run
```bash
cd path/to/pmc-data-extraction
pip install --upgrade pip
poetry install --no-root --with test --all-extras
```
The above command assumes that you would install `mmlearn` or `open_clip` packages from source using the submodules found in `pmc-data-extraction/openpmcvl/`experiment.

3. Clone `mmlearn` and `open_clip` submodules.
```bash
git submodule init
git submodule update
```
You should see the source files inside `pmc-data-extraction/openpmcvl/experiment/open_clip` and `pmc-data-extraction/openpmcvl/experiment/mmlearn`.

4. Install `mmlearn` from source.
```bash
cd openpmcvl/experiment/mmlearn
python3 -m pip install -e .
```

5. Install `open_clip` from source.
```bash
cd ../open_clip
make install
make install-training
```

6. Check installations.
```bash
pip freeze | grep mmlearn
pip freeze | grep open_clip
python
> import mmlearn
> import open_clip
> mmlearn.__file__
> open_clip.__file__
```

**Note:** Since these submodules (`mmlearn` and `open_clip`) are only part of the main branch in a single repository, if you change your branch to a branch where these submodules don't exist, your python interpretor won't be able to find these packages and you will face errors.


## Download and parse image-caption pairs from Pubmed Articles
The codebase used to download Pubmed articles and parse image-text pairs from them is stored in `openpmcvl/foundation`.
This codebase heavily relies on [Build PMC-OA](https://github.com/WeixiongLin/Build-PMC-OA) codebase[[1]](#1).
To download and parse articles with licenses that allow commercial use, run
```bash
# activate virtual environment
source /path/to/your/venv/bin/activate
# navigate to root directory of the package
cd openpmcvl/foundation
# download all 11 volumes with commercailly usable license
python -u src/fetch_oa.py --num-retries 5 --extraction-dir path/to/download/directory/commercial --license-type comm --volumes 0 1 2 3 4 5 6 7 8 9 10 11
```
To download and parse open-access articles which are not allowed commercial use, run
```bash
python -u src/fetch_oa.py --num-retries 5 --extraction-dir path/to/download/directory/noncommercial --license-type noncomm --volumes 1 2 3 4 5 6 7 8 9 10 11
```
To download and parse open-access articles which other licenses than what is mentioned above, run
```bash
python -u src/fetch_oa.py --num-retries 5 --extraction-dir path/to/download/directory/other --license-type other --volumes 0 1 2 3 4 5 6 7 8 9 10 11
```


## Run Benchmarking Experiments
We use `mmlearn` to run benchmarking experiments.
Many experiments can be run with our dataset and `mmlearn`.
A simple example of training with our dataset is given below:
```bash
# navigate to root directory of the repository
cd pmc-data-extraction
# set pythonpath
export PYTHONPATH="./"
# run training experiment
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched \
    experiment_name=pmcoa2_matched_train \
    dataloader.train.batch_size=256 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False
```

Four downstream evaluation experiments can be run with checkpoints generated during training: cross-modal retrieval, zero-shot classification, linear probing, and patient-to-patient retrieval.
An example of cross-modal retrieval on the MIMIC-IV-CXR dataset is given below:
```bash
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched \
    experiment_name=pmcoa2_matched_retrieval_mimic \
    job_type=eval \
    ~datasets.test.pmcoa2 \
    +datasets@datasets.test.mimic=MIMICIVCXR \
    datasets.test.mimic.split=test \
    +datasets/transforms@datasets.test.mimic.transform=biomedclip_vision_transform \
    datasets.test.mimic.transform.job_type=eval \
    dataloader.test.batch_size=64 \
    resume_from_checkpoint="path/to/model/checkpoint"
```
For more comprehensive examples of shell scripts that run various experiments with Open-PMC, refer to `openpmcvl/experiment/scripts`.
For more information about `mmlearn`, please refer to the package's [official codebase](https://github.com/VectorInstitute/mmlearn).


## Citation
If you find the code useful for your research, please consider citing
```bib
@article{baghbanzadeh2025advancing,
  title={Advancing Medical Representation Learning Through High-Quality Data},
  author={Baghbanzadeh, Negin and Fallahpour, Adibvafa and Parhizkar, Yasaman and Ogidi, Franklin and Roy, Shuvendu and Ashkezari, Sajad and Khazaie, Vahid Reza and Colacci, Michael and Etemad, Ali and Afkanpour, Arash and Dolatabadi, Elham},
  journal={arXiv preprint arXiv:2503.14377},
  year={2025}
}
```
