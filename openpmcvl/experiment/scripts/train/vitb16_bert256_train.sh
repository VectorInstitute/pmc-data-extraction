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
    +experiment=biomedclip_matched \
    experiment_name=vitb16_bert256_train_bs256_matched \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.T_max=104671 \
    task.lr_scheduler.scheduler.warmup_length=2000


# test
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
    task.lr_scheduler.scheduler.T_max=50 \
    task.lr_scheduler.scheduler.warmup_length=10 \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.log_every_n_steps=1
