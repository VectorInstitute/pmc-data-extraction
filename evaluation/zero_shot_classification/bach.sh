#!/bin/bash
# Zero-shot image classification on BACH (breast histology).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set BACH_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   BACH_ROOT_DIR=/path/to/bach bash bach.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_bach \
    job_type=eval \
    +datasets@datasets.test.bach=BACH \
    datasets.test.bach.split=test \
    +datasets/transforms@datasets.test.bach.transform=biomedclip_vision_transform \
    datasets.test.bach.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
