# on slurm
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
    experiment_name=vitb16_bert256_train_bs256_matched_nw3 \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=3 \
    dataloader.val.num_workers=3 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.t_max=104671 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13674847/last.ckpt" \
    trainer.logger.wandb.id="0e2xm2bk" \
    trainer.logger.wandb.resume="must"
