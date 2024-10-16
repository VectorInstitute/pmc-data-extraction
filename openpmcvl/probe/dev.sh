# Parameters to be set in classifier.py
# keywords = None
# templates = None
# gt_labels = False
# include_entry = True
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=pmcvl_mc \
    datasets.test.pmcvl.include_entry=True \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint=""


# load another dataset - LC25000 colon
## SET include_text to False in classifier.py
## SET include_entry to False in classifier.py
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=lc25000_mc \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.lc25000=LC25000 \
    datasets.test.lc25000.root_dir=${LC25000_COLON_ROOT_DIR} \
    datasets.test.lc25000.organ=colon \
    +datasets/transforms@datasets.test.lc25000.transform=biomedclip_vision_transform \
    datasets.test.lc25000.split=test \
    datasets.test.lc25000.transform.job_type=eval \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True
