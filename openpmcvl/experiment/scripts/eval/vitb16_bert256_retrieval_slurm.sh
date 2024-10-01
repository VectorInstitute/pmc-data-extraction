mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=long \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=16 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=2880 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=biomedclip_retrieval_matched_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_clean_1 \
    dataloader.test.batch_size=16 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar//last.ckpt"

# comment: test_clean_1 is an experimental split with 400K pairs.
