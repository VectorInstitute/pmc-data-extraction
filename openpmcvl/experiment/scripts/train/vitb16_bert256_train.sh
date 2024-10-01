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


# test retrieval
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_train_bs8_test \
    job_type=eval \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1 \
    > outputs_bytes.txt

# test retrieval on slurm 
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=1440 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_retrieval \
    experiment_name=vitb16_bert256_retrieval_test \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_clean \
    trainer.logger.wandb.offline=True
    
