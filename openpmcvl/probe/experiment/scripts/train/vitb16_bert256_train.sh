# local
mmlearn_run \
    'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=vitb16_bert256_train_bs64 \
    dataloader.train.batch_size=64 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=2 \
    dataloader.val.num_workers=2 \
    datasets/transforms@datasets.train.pmcvl.transform=med_clip_vision_transform \
    datasets/transforms@datasets.val.pmcvl.transform=med_clip_vision_transform \
    datasets/transforms@datasets.test.pmcvl.transform=med_clip_vision_transform \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13578302/last.ckpt" \
    trainer.logger.wandb.id="8p6lnk48" \
    trainer.logger.wandb.resume="must"
