# local
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=vitb16_bert256_train_bs8_test \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.lr_scheduler.scheduler.T_max=104671 \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13578302/last.ckpt" \
    trainer.logger.wandb.id="8p6lnk48" \
    trainer.logger.wandb.resume="must"

# test
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_train_bs8_test \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.T_max=104671 \
    task.lr_scheduler.scheduler.warmup_length=2000


# test
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_train_bs8_test \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.lr_scheduler.scheduler.T_max=50 \
    task.lr_scheduler.scheduler.warmup_length=10 \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1


# test eval
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_train_bs8_test \
    job_type=eval \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.lr_scheduler.scheduler.T_max=50 \
    task.lr_scheduler.scheduler.warmup_length=10 \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1

# test eval on slurm 
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_train_bs8_test \
    job_type=eval \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.lr_scheduler.scheduler.T_max=50 \
    task.lr_scheduler.scheduler.warmup_length=10 \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1
    
