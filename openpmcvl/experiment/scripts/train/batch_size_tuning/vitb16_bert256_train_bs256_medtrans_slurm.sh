# on slurm
# note: T_max is computed wrong here.
# note: biomedclip config does not exactly match
# the experiment run in biomedclip's original paper.
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
    'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=vitb16_bert256_train_bs256 \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    datasets/transforms@datasets.train.pmcvl.transform=med_clip_vision_transform \
    datasets/transforms@datasets.val.pmcvl.transform=med_clip_vision_transform \
    datasets/transforms@datasets.test.pmcvl.transform=med_clip_vision_transform \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.T_max=107555 \
    strict_loading=False \
    resume_from_checkpoint="path/to/checkpoint" \
    trainer.logger.wandb.id="60k51nv8" \
    trainer.logger.wandb.resume="must"
