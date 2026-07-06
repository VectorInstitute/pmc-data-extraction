#!/bin/bash
# Zero-shot cross-modal (image <-> text) retrieval on DeepEyeNet (test split).
#
# By default this evaluates the released Open-PMC-18M checkpoint from the Hugging Face Hub
# (vector-institute/open-pmc-18m-clip), downloaded automatically. To evaluate a different
# checkpoint, set CKPT to a local open_clip .pt/.bin file:
#   CKPT=/path/to/checkpoint.pt bash deepeyenet.sh
set -euo pipefail
CKPT="${CKPT:-hf-hub:vector-institute/open-pmc-18m-clip}"

mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_localckpt_retrieval \
    experiment_name=open_pmc_18m_retrieval_dey \
    job_type=eval \
    +datasets@datasets.test.dey=DeepEyeNet \
    datasets.test.dey.split=test \
    +datasets/transforms@datasets.test.dey.transform=biomedclip_vision_transform \
    datasets.test.dey.transform.job_type=eval \
    task.encoders.text.checkpoint_path="$CKPT" \
    task.encoders.rgb.checkpoint_path="$CKPT" \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k='[10,50,200]' \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k='[10,50,200]' \
    ~task.postprocessors.norm_and_logit_scale.logit_scale \
    ~task.postprocessors.norm_and_logit_scale.norm
