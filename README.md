<div align="center">

# Open-PMC

**Large-scale medical vision–language pretraining from PubMed Central**

If you find this project useful, please give us a star 🌟.

<a href="https://arxiv.org/abs/2503.14377"><img src="https://img.shields.io/badge/Open--PMC-arXiv-b31b1b"></a>
<a href="https://arxiv.org/abs/2506.02738"><img src="https://img.shields.io/badge/Open--PMC--18M-arXiv-b31b1b"></a>
<a href="https://huggingface.co/vector-institute/open-pmc-18m-clip"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Open--PMC--18M-blue"></a>
<a href="https://huggingface.co/datasets/vector-institute/open-pmc-18m"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Open--PMC--18M-yellow"></a>
<a href="https://www.youtube.com/watch?v=6XNclnlT90I"><img src="https://img.shields.io/badge/Video-MICCAI%202025%20Oral-red?logo=youtube&logoColor=white"></a>
<a href="https://github.com/VectorInstitute/pmc-data-extraction/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/License-Apache%202.0-green"></a>

</div>

<div align="center">
    <img src="https://raw.githubusercontent.com/VectorInstitute/pmc-data-extraction/0a969136344a07267bb558d01f3fe76b36b93e1a/media/open-pmc-pipeline.png" alt="Open-PMC Pipeline" width="1000" />
</div>

Open-PMC is a toolkit for **training and evaluating** CLIP-style medical vision–language models on
large-scale image–text pairs mined from open-access PubMed Central articles. It spans the full
pipeline: downloading and parsing figure–caption pairs, contrastive pretraining with
[`mmlearn`](https://github.com/VectorInstitute/mmlearn), and a **zero-shot evaluation** suite.

**Evaluation** measures how well the learned image and text embeddings align, with no task-specific
fine-tuning:

- **Zero-shot cross-modal retrieval** — use a caption to rank images (and an image to rank captions)
  and report Recall@K, on Quilt-1M, MIMIC-IV-CXR, and DeepEyeNet.
- **Zero-shot classification** — label an image by matching it against text prompts built from each
  class name (top-1 accuracy), across pathology, dermatology, radiology, and MedMNIST+ datasets.

This repository hosts the code for **Open-PMC**
([arXiv:2503.14377](https://arxiv.org/abs/2503.14377), MICCAI 2025 **oral**) and **Open-PMC-18M**
([arXiv:2506.02738](https://arxiv.org/abs/2506.02738), MICCAI 2026).

## News

- [x] **`Jul 2026.`** Released **Open-PMC-18M** — the OpenCLIP checkpoint, models, and dataset are in the [🤗 Open-PMC-18M collection](https://huggingface.co/collections/vector-institute/open-pmc-18m).
- [x] **`May 2026.`** **Open-PMC-18M** has been accepted at **MICCAI 2026**! 🎉
- [x] **`Sep 2025.`** **Open-PMC** was presented as an **oral** at MICCAI 2025 — [watch the talk ▶️](https://www.youtube.com/watch?v=6XNclnlT90I).
- [x] **`Jun 2025.`** **Open-PMC-18M** is on [arXiv](https://arxiv.org/abs/2506.02738).
- [x] **`May 2025.`** **Open-PMC** has been accepted as an **oral** at **MICCAI 2025**! 🎉
- [x] **`Mar 2025.`** **Open-PMC** is on [arXiv](https://arxiv.org/abs/2503.14377) — models and dataset are in the [🤗 OpenPMC collection](https://huggingface.co/collections/vector-institute/openpmc).

## Table of Contents

1. [Installing Dependencies](#installing-dependencies)
2. [Benchmarking](#benchmarking)
3. [Evaluation](#evaluation)
4. [Results](#results)
5. [Citation](#citation)

## Installing dependencies

We use [poetry](https://python-poetry.org/docs/#installation) for dependency management.

```bash
# 1. Create and activate a Python 3.10 environment
python -m venv .venv && source .venv/bin/activate

# 2. Install the package with mmlearn + open_clip (from pip)
cd path/to/pmc-data-extraction
pip install --upgrade pip
poetry install --no-root --with test,open_clip,mmlearn --all-extras
```

Prefer building [`mmlearn`](https://github.com/VectorInstitute/mmlearn) and
[`open_clip`](https://github.com/mlfoundations/open_clip) from source? Install without them
(`poetry install --no-root --with test --all-extras`), then pull the submodules and install them:

```bash
git submodule update --init
pip install -e openpmcvl/experiment/mmlearn
cd openpmcvl/experiment/open_clip && make install && make install-training
```

Verify with `python -c "import mmlearn, open_clip; print(mmlearn.__file__, open_clip.__file__)"`.

## Benchmarking

We use [`mmlearn`](https://github.com/VectorInstitute/mmlearn) to run training and evaluation.
A minimal training run:

```bash
cd pmc-data-extraction
export PYTHONPATH="./"
mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched \
    experiment_name=pmcoa2_matched_train \
    dataloader.train.batch_size=256 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False
```

Additional training/eval shell scripts are under `openpmcvl/experiment/scripts`.

## Evaluation

Ready-to-run **zero-shot retrieval** and **zero-shot classification** scripts live in
[`evaluation/`](evaluation/). By default they evaluate the released
[Open-PMC-18M](https://huggingface.co/vector-institute/open-pmc-18m-clip) checkpoint, downloaded
automatically from the Hugging Face Hub — just set the dataset's root directory:

```bash
# cross-modal retrieval
QUILT_ROOT_DIR=/data/quilt bash evaluation/zero_shot_retrieval/quilt.sh

# zero-shot classification
PCAM_ROOT_DIR=/data/pcam bash evaluation/zero_shot_classification/pcam.sh
```

To evaluate a different checkpoint, set `CKPT` to a local open_clip `.pt`/`.bin` file. See
[`evaluation/README.md`](evaluation/README.md) for the full list of datasets and options.

## Results

Zero-shot cross-modal retrieval with **Open-PMC-18M** (Recall@200), produced by the scripts in
[`evaluation/zero_shot_retrieval/`](evaluation/zero_shot_retrieval/):

| Dataset             | Image→Text R@200 | Text→Image R@200 |
|---------------------|:----------------:|:----------------:|
| Quilt-1M (val)      |      25.53%      |      27.16%      |
| MIMIC-IV-CXR (test) |      27.47%      |      28.14%      |
| DeepEyeNet (test)   |      19.30%      |      20.48%      |

<sub>Reproduce with e.g. `QUILT_ROOT_DIR=… bash evaluation/zero_shot_retrieval/quilt.sh`.</sub>

## Citation

If you find this code useful for your research, please consider citing:

```bib
@article{baghbanzadeh2025advancing,
  title={Advancing Medical Representation Learning Through High-Quality Data},
  author={Baghbanzadeh, Negin and Fallahpour, Adibvafa and Parhizkar, Yasaman and Ogidi, Franklin and Roy, Shuvendu and Ashkezari, Sajad and Khazaie, Vahid Reza and Colacci, Michael and Etemad, Ali and Afkanpour, Arash and Dolatabadi, Elham},
  journal={arXiv preprint arXiv:2503.14377},
  year={2025}
}

@article{baghbanzadeh2025open,
  title={Open-pmc-18m: A high-fidelity large scale medical dataset for multimodal representation learning},
  author={Baghbanzadeh, Negin and Islam, Mohammed Saidul and Ashkezari, Sajad and Dolatabadi, Elham and Afkanpour, Arash},
  journal={arXiv preprint arXiv:2506.02738},
  year={2025}
}
```
