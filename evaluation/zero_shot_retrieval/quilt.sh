#!/bin/bash
# Zero-shot cross-modal (image <-> text) retrieval on Quilt-1M (val split).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. To evaluate a different
# checkpoint, set CKPT to a local open_clip .pt/.bin file:
#   CKPT=/path/to/checkpoint.pt bash quilt.sh
set -euo pipefail
# Keep temp files on node-local disk to avoid NFS ".nfs* busy" cleanup errors on clusters.
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_retrieval \
    experiment_name=open_pmc_18m_retrieval_quilt \
    job_type=eval \
    +datasets@datasets.test.quilt=Quilt \
    datasets.test.quilt.split=val \
    +datasets/transforms@datasets.test.quilt.transform=biomedclip_vision_transform \
    datasets.test.quilt.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k='[10,50,200]' \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k='[10,50,200]' \
    ~task.postprocessors.norm_and_logit_scale.logit_scale \
    ~task.postprocessors.norm_and_logit_scale.norm
