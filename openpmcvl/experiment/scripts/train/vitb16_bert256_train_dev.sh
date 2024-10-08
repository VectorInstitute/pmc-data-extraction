# example (local)
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
    task.lr_scheduler.scheduler.t_max=104671 \
    strict_loading=False \
    resume_from_checkpoint="path/to/checkpoint" \
    trainer.logger.wandb.id="8p6lnk48" \
    trainer.logger.wandb.resume="must"

# development pmcvl
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=vitb16_bert256_train_bs8_dev \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.lr_scheduler.scheduler.t_max=50 \
    task.lr_scheduler.scheduler.warmup_length=10 \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1

# development pmcoa
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=vitb16_bert256_pmcoa \
    experiment_name=vitb16_bert256_train_bs8_dev \
    job_type=train \
    dataloader.train.batch_size=8 \
    dataloader.val.batch_size=8 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    trainer.logger.wandb.offline=True \
    trainer.log_every_n_steps=1
