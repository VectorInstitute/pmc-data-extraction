# create a directory for checkpoints on scratch
mkdir /home/yaspar/${oc.env:USER}/${oc.env:SLURM_JOB_ID}

mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \