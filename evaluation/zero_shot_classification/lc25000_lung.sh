#!/bin/bash
# Zero-shot image classification on LC25000 (lung histology).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set LC25000_LUNG_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   LC25000_LUNG_ROOT_DIR=/path/to/lc25k_lung bash lc25000_lung.sh
set -euo pipefail
# Keep temp files on node-local disk to avoid NFS ".nfs* busy" cleanup errors on clusters.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_lc25k_lung \
    job_type=eval \
    +datasets@datasets.test.lc25k_lung=LC25000 \
    datasets.test.lc25k_lung.split=test \
    datasets.test.lc25k_lung.organ=lung \
    +datasets/transforms@datasets.test.lc25k_lung.transform=biomedclip_vision_transform \
    datasets.test.lc25k_lung.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
