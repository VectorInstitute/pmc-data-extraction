# original
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=vitb16_bert256_pmcoa \
    experiment_name=vitb16_bert256_pmcoa_retrieval_deepeyenet \
    job_type=eval \
    ~datasets.test.pmcoa \
    +datasets@datasets.test.dey=DeepEyeNet \
    datasets.test.dey.split=test \
    +datasets/transforms@datasets.test.dey.transform=biomedclip_vision_transform \
    datasets.test.dey.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=256 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=False \
    resume_from_checkpoint=""


# neurips hps
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=vitb16_bert256_pmcoa_neurips_hps \
    experiment_name=vitb16_bert256_pmcoa_neurips_hps_retrieval_deepeyenet \
    job_type=eval \
    ~datasets.test.pmcoa \
    +datasets@datasets.test.dey=DeepEyeNet \
    datasets.test.dey.split=test \
    +datasets/transforms@datasets.test.dey.transform=med_clip_vision_transform \
    datasets.test.dey.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=256 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/pmc_oa/neurips_hps/last.ckpt"