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
    +experiment=vitb16_gpt77_pmcoa \
    experiment_name=vitb16_gpt77_pmcoa_retrieval_pmcvl \
    job_type=eval \
    ~datasets.test.pmcoa \
    +datasets@datasets.test.pmcvl=PMCVL \
    datasets.test.pmcvl.split=test_cleaner \
    +datasets/transforms@datasets.test.pmcvl.transform=biomedclip_vision_transform \
    datasets.test.pmcvl.transform.job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=""
