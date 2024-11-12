mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmc_clip \
    experiment_name=pmc_clip_retrieval_quilt \
    job_type=eval \
    ~datasets.test.pmcoa2 \
    +datasets@datasets.test.quilt=Quilt \
    datasets.test.quilt.split=val \
    +datasets/transforms@datasets.test.quilt.transform=pmc_clip_vision_transform \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=PmcClipTokenizer \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.postprocessors.norm_and_logit_scale.logit_scale.logit_scale_init=4.4292 \
    task.postprocessors.norm_and_logit_scale.logit_scale.learnable=False \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    ~task.postprocessors.norm_and_logit_scale.norm \
    strict_loading=True \
    resume_from_checkpoint=""
