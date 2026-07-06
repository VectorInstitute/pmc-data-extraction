#!/bin/bash
# Zero-shot image classification on SICAPv2 (prostate).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set SICAP_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   SICAP_ROOT_DIR=/path/to/sicap bash sicap.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_sicap \
    job_type=eval \
    +datasets@datasets.test.sicap=SICAP \
    datasets.test.sicap.split=test \
    +datasets/transforms@datasets.test.sicap.transform=biomedclip_vision_transform \
    datasets.test.sicap.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
