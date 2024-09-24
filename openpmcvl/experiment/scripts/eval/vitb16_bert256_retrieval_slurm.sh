mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_baseline_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_clean_1 \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13571189/last.ckpt"

# comment: test_clean_1 is an experimental split with 400K pairs.
