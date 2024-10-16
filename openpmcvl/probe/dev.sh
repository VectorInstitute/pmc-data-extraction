python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=modality_classification_test \
    datasets.test.pmcvl.include_entry=True \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint=""


# load another dataset
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=modality_classification_test \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.lc25000=LC25000 \
    +datasets/transforms@datasets.test.lc25000.transform=biomedclip_vision_transform \
    datasets.test.lc25000.split=test \
    datasets.test.lc25000.transform.job_type=eval \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True
