# local evaluations
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_1pair \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.patient_q.pretrained=True \
    task.encoders.patient_t.pretrained=True \
    trainer.logger.wandb.offline=True
