python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=modality_classification_test \
    datasets.test.pmcvl.include_entry=True \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint=""
