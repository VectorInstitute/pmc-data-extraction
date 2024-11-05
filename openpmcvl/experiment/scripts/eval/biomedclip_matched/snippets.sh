# header to run experiments on rtx6000
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=long \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=32 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=2880


# config to turn wandb offline
    trainer.logger.wandb.offline=True

# config to resume a wandb run
    trainer.logger.wandb.resume="must" \
    trainer.logger.wandb.id=""  # find wandb run id at the end of the run's url
