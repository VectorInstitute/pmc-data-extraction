mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a100_arashaf \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
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
    dataloader.test.num_workers=2 \
    'task.evaluation_tasks.retrieval.task.task_specs=[{query_modality:rgb,target_modality:text,top_k:[1,5,10]}]' \
    strict_loading=False \
    resume_from_checkpoint=""

# comment: test_clean_1 is an experimental split with 400K pairs.
