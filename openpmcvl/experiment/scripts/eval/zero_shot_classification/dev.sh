mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=modality_classification_test \
    job_type=eval \
    ~task.evaluation_tasks.retrieval \
    datasets.test.pmcvl.include_entry=True \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True
