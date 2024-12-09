# train on a40 biomedclip
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=420 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_train \
    dataloader.train.batch_size=32 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.text.clip_ckpt=""  \
    task.optimizer.lr=1e-6 \
    task.lr_scheduler.scheduler.t_max=10053 \
    task.lr_scheduler.scheduler.warmup_length=4000 \
    ~trainer.callbacks.early_stopping

# eval on rtx6000 biomedclip
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a100_arashaf \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=20 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_eval \
    job_type=eval \
    dataloader.test.batch_size=128 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.text.clip_ckpt="" \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[1000] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[1000]
