#!/bin/bash
# Zero-shot image classification on a MedMNIST+ variant (224px).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. Set CKPT to a local
# open_clip .pt/.bin file to evaluate a different checkpoint.
#
# Set MEDMNISTPLUS_ROOT_DIR to a dir with the MedMNIST+ 224px files (e.g. pathmnist_224.npz).
#
# Usage (NAME selects the variant; defaults to pathmnist):
#   NAME=pathmnist MEDMNISTPLUS_ROOT_DIR=/data/medmnist bash medmnist.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"
NAME="${NAME:-pathmnist}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_ZSC \
    experiment_name=open_pmc_18m_zsc_medmnist_${NAME} \
    job_type=eval \
    +datasets@datasets.test.medmnist=MedMNISTPlus \
    datasets.test.medmnist.name=${NAME} \
    datasets.test.medmnist.split=test \
    +datasets/transforms@datasets.test.medmnist.transform=biomedclip_vision_transform \
    datasets.test.medmnist.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4
