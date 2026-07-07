# Evaluation

Zero-shot evaluation of an open_clip-format checkpoint with the BiomedCLIP architecture
(ViT-B/16 image encoder + PubMedBERT text encoder), such as the **Open-PMC-18M** model:
[vector-institute/open-pmc-18m-clip](https://huggingface.co/vector-institute/open-pmc-18m-clip).

```
evaluation/
├── zero_shot_retrieval/         # image ↔ text cross-modal retrieval
│   ├── quilt.sh                 # Quilt-1M
│   ├── mimic.sh                 # MIMIC-IV-CXR
│   └── deepeyenet.sh            # DeepEyeNet
└── zero_shot_classification/    # zero-shot image classification
    ├── pcam.sh                  # PatchCamelyon
    ├── bach.sh                  # BACH (breast histology)
    ├── sicap.sh                 # SICAPv2 (prostate)
    ├── nck_crc.sh               # NCT-CRC-HE (colorectal)
    ├── pad_ufes_20.sh           # PAD-UFES-20 (skin lesions)
    ├── ham10000.sh              # HAM10000 (skin lesions)
    ├── lc25000_lung.sh          # LC25000 (lung histology)
    └── medmnist.sh              # MedMNIST+ variants (pathmnist, dermamnist, ...)
```

Every script loads the checkpoint through the encoders' `checkpoint_path` argument (an
open_clip-native `state_dict` with `visual.*` / `text.*` keys) — no Lightning
`resume_from_checkpoint` conversion needed. Retrieval uses the `biomedclip_localckpt_retrieval`
config; classification uses `biomedclip_localckpt_ZSC`.

## 1. Setup

Install the repo (see the [top-level README](../README.md)). **No checkpoint setup is needed** —
by default the scripts evaluate the released **Open-PMC-18M** checkpoint
([vector-institute/open-pmc-18m-clip](https://huggingface.co/vector-institute/open-pmc-18m-clip)),
which is downloaded automatically from the Hugging Face Hub on first run.

To evaluate a **different** checkpoint instead, set `CKPT` to a local open_clip `.pt`/`.bin` file
(or another `hf-hub:<org>/<repo>` reference).

## 2. Run

**Every dataset reads its location from a `*_ROOT_DIR` environment variable** (see
[Data setup](#3-data-setup) below). If a `*_ROOT_DIR` is not set, the run fails with a Hydra
*"missing mandatory value"* error for `root_dir`.

```bash
# retrieval — evaluates the released Open-PMC-18M checkpoint by default
QUILT_ROOT_DIR=/data/quilt bash evaluation/zero_shot_retrieval/quilt.sh

# classification
PCAM_ROOT_DIR=/data/pcam bash evaluation/zero_shot_classification/pcam.sh
NAME=pathmnist MEDMNISTPLUS_ROOT_DIR=/data/medmnist bash evaluation/zero_shot_classification/medmnist.sh

# to evaluate your own checkpoint instead, set CKPT to a local file
CKPT=/path/to/checkpoint.pt QUILT_ROOT_DIR=/data/quilt bash evaluation/zero_shot_retrieval/quilt.sh
```

Retrieval reports Recall@{10, 50, 200} in both directions; classification reports top-1 accuracy.

## 3. Data setup

Set the listed `*_ROOT_DIR` before running. Datasets marked **auto** download themselves from the
Hugging Face Hub into that directory (make it writable); datasets marked **manual** must be
downloaded from the source first and arranged as shown.

| Script            | `*_ROOT_DIR`            | Source                                  | Download |
|-------------------|-------------------------|-----------------------------------------|----------|
| `quilt.sh`        | `QUILT_ROOT_DIR`        | [Quilt-1M](https://github.com/wisdomikezogwo/quilt1m) | manual |
| `mimic.sh`        | `MIMICIVCXR_ROOT_DIR`   | [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) (credentialed) | manual |
| `deepeyenet.sh`   | `DEY_ROOT_DIR`          | [DeepEyeNet](https://github.com/Jhhuangkay/DeepOpht-Medical-Report-Generation-for-Retinal-Images-via-Deep-Models-and-Visual-Explanation) | manual |
| `pcam.sh`         | `PCAM_ROOT_DIR`         | HF `1aurent/PatchCamelyon`              | auto |
| `bach.sh`         | `BACH_ROOT_DIR`         | HF `1aurent/BACH`                       | auto |
| `nck_crc.sh`      | `NCK_CRC_ROOT_DIR`      | HF `DykeF/NCTCRCHE100K`                 | auto |
| `sicap.sh`        | `SICAP_ROOT_DIR`        | [SICAPv2](https://data.mendeley.com/datasets/9xxm58dvs3/1) | manual |
| `pad_ufes_20.sh`  | `PADUFES_ROOT_DIR`      | [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | manual |
| `ham10000.sh`     | `HAM10000_ROOT_DIR`     | [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) | manual |
| `lc25000_lung.sh` | `LC25000_LUNG_ROOT_DIR` (and `LC25000_COLON_ROOT_DIR` for colon) | [LC25000](https://github.com/tampapath/lung_colon_image_set) | manual (pre-build) |
| `medmnist.sh`     | `MEDMNISTPLUS_ROOT_DIR` | [MedMNIST+](https://medmnist.com/) (224px)      | manual |

### Expected layout for the manual datasets

**Quilt-1M** — `$QUILT_ROOT_DIR/`
```
quilt_1m_val.csv          # split metadata (also quilt_1m_train.csv)
quilt_1m/                 # image files referenced by image_path in the csv
```

**MIMIC-IV-CXR** — `$MIMICIVCXR_ROOT_DIR/` — the split metadata file (`.json`/`.csv`) plus the
CXR image tree. Requires credentialed PhysioNet access.

**DeepEyeNet** — `$DEY_ROOT_DIR/`
```
DeepEyeNet_test.json      # split file (also _train.json / _val.json)
<image folders>           # image paths referenced by the json keys
```

**SICAPv2** — `$SICAP_ROOT_DIR/`
```
images/                   # patch images
partition/Test/Train.xlsx
partition/Test/Test.xlsx  # columns: image_name, NC, G3, G4, G5
```

**PAD-UFES-20** — `$PADUFES_ROOT_DIR/`
```
metadata.csv              # columns: img_id, diagnostic
Dataset/                  # image files named by img_id
```

**HAM10000** — `$HAM10000_ROOT_DIR/`
```
HAM10000_metadata.csv     # train/test csvs are derived from this on first run
skin_cancer/              # <image_id>.jpg files
```

**LC25000** — pre-build a 🤗 `datasets` arrow directory per organ/split:
`$LC25000_LUNG_ROOT_DIR/cache/lc25000_lung_test.arrow` (and `..._colon_test.arrow` under
`$LC25000_COLON_ROOT_DIR`). Each record needs `image` and `label` fields.

**MedMNIST+** — `$MEDMNISTPLUS_ROOT_DIR/` with the 224px npz files, e.g. `pathmnist_224.npz`,
`dermamnist_224.npz`, `bloodmnist_224.npz`, `organamnist_224.npz`, … Select one with `NAME=`.

**Auto datasets** (PCam, BACH, NCT-CRC-HE) need only a writable `*_ROOT_DIR`; on first run they
download into `<root>/scratch/` and cache into `<root>/cache/`.

## Notes

- **Different checkpoint / dataset.** Point `CKPT` at any open_clip-format checkpoint with the
  BiomedCLIP architecture. To evaluate another retrieval dataset, copy a script and swap the
  dataset name (e.g. `+datasets@datasets.test.<key>=<DatasetClass>`).
- **SLURM.** The scripts run a single process on the local GPU. To submit to a cluster, wrap the
  `mmlearn_run` call with `--multirun` and the hydra submitit launcher, or call the script from
  your `sbatch` wrapper.
