# with ROCO dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_roco \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.roco=ROCO \
    datasets.test.roco.split=test \
    +datasets/transforms@datasets.test.roco.transform=med_clip_vision_transform \
    datasets.test.roco.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=0 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint=""
