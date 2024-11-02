mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_cleaner \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.postprocessors.norm_and_logit_scale.logit_scale.logit_scale_init=4.4454 \
    task.postprocessors.norm_and_logit_scale.logit_scale.learnable=False \
    ~task.postprocessors.norm_and_logit_scale.norm \
    strict_loading=False \
    resume_from_checkpoint=""

# comment: test_clean_1 is an experimental split with 400K pairs.
