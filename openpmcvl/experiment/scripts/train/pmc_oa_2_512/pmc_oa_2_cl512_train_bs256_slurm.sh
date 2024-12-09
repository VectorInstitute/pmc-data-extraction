# a40
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
    +experiment=pmcoa2_matched_512 \
    experiment_name=pmcoa2_matched_512_train \
    dataloader.train.batch_size=128 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    trainer.max_epochs=64 \
    task.lr_scheduler.scheduler.t_max=54476 \
    task.lr_scheduler.scheduler.warmup_length=5448 \
    ~trainer.callbacks.early_stopping

# a100
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a100_arashaf \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched_512 \
    experiment_name=pmcoa2_matched_512_train \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=3 \
    dataloader.val.num_workers=3 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    trainer.max_epochs=64 \
    task.lr_scheduler.scheduler.t_max=27238 \
    task.lr_scheduler.scheduler.warmup_length=2724 \
    ~trainer.callbacks.early_stopping
