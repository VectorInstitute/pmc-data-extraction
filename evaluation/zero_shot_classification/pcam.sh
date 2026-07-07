#!/bin/bash
# Zero-shot image classification on PatchCamelyon (PCam).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set PCAM_ROOT_DIR to the dataset root directory (see evaluation/README.md).
#
# Usage:
#   PCAM_ROOT_DIR=/path/to/pcam bash pcam.sh
set -euo pipefail
# Keep temp files on node-local disk to avoid NFS ".nfs* busy" cleanup errors on clusters.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_pcam \
    job_type=eval \
    +datasets@datasets.test.pcam=PCAM \
    +datasets/transforms@datasets.test.pcam.transform=biomedclip_vision_transform \
    datasets.test.pcam.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
