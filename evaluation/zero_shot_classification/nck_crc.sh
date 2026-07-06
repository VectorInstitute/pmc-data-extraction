#!/bin/bash
# Zero-shot image classification on NCT-CRC-HE (colorectal).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set NCK_CRC_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   NCK_CRC_ROOT_DIR=/path/to/nck_crc bash nck_crc.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_nck_crc \
    job_type=eval \
    +datasets@datasets.test.nck_crc=NckCrc \
    datasets.test.nck_crc.split=validation \
    +datasets/transforms@datasets.test.nck_crc.transform=biomedclip_vision_transform \
    datasets.test.nck_crc.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
