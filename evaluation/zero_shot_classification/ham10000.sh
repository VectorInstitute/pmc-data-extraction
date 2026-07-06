#!/bin/bash
# Zero-shot image classification on HAM10000 (skin lesions).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set HAM10000_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   HAM10000_ROOT_DIR=/path/to/ham10k bash ham10000.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_ham10k \
    job_type=eval \
    +datasets@datasets.test.ham10k=HAM10000 \
    datasets.test.ham10k.split=test \
    +datasets/transforms@datasets.test.ham10k.transform=biomedclip_vision_transform \
    datasets.test.ham10k.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
