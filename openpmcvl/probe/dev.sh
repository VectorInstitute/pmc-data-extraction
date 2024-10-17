# for openpmcvl dataset
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_matched_mc_pmcvl \
    datasets.test.pmcvl.include_entry=True \
    dataloader.test.batch_size=4 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint=""

# for lc25000
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_matched_mc_lc25000_lung \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.lc25000=LC25000 \
    datasets.test.lc25000.root_dir=${LC25000_LUNG_ROOT_DIR} \
    datasets.test.lc25000.organ=lung \
    +datasets/transforms@datasets.test.lc25000.transform=biomedclip_vision_transform \
    datasets.test.lc25000.split=test \
    datasets.test.lc25000.transform.job_type=eval \
    dataloader.test.batch_size=32 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch18-step62149.ckpt"

# for lc25000 with med_benchmarking checkpoint
python openpmcvl/probe/classifier.py \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=med_benchmarking \
    experiment_name=med_benchmarking_mc_lc25000_lung \
    +datasets@datasets.test.lc25000=LC25000 \
    datasets.test.lc25000.root_dir=${LC25000_LUNG_ROOT_DIR} \
    datasets.test.lc25000.organ=lung \
    +datasets/transforms@datasets.test.lc25000.transform=biomedclip_vision_transform \
    +datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=HFCLIPTokenizer \
    datasets.test.lc25000.split=test \
    datasets.test.lc25000.transform.job_type=eval \
    dataloader.test.batch_size=32 \
    trainer.logger.wandb.offline=True \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt"