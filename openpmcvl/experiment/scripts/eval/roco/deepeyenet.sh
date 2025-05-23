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
    +experiment=roco_matched \
    experiment_name=roco_matched_retrieval_deepeyenet \
    job_type=eval \
    ~datasets.test.roco \
    +datasets@datasets.test.dey=DeepEyeNet \
    datasets.test.dey.split=test \
    +datasets/transforms@datasets.test.dey.transform=biomedclip_vision_transform \
    datasets.test.dey.transform.job_type=eval \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=True \
    resume_from_checkpoint=""
