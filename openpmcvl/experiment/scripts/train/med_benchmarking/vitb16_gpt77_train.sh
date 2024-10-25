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
    +experiment=med_benchmarking \
    experiment_name=med_benchmarking_train \
    ~trainer.callbacks.early_stopping \
    strict_loading=True \
    resume_from_checkpoint="path/to/checkpoint" \
    trainer.logger.wandb.id="" \
    trainer.logger.wandb.resume="must"
