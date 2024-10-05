# on slurm
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
    +experiment=biomedclip_matched \
    experiment_name=vitb16_bert256_train_lr2e-5_matched \
    dataloader.train.batch_size=128 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.optimizer.lr=2.0e-5 \
    task.optimizer.weight_decay=0.2\
    task.lr_scheduler.scheduler.t_max=209342 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13699479/last.ckpt" \
    trainer.logger.wandb.id="my1mre6k" \
    trainer.logger.wandb.resume="must"
