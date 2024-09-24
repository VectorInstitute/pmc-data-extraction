# with OpenPMC-VL dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_baseline_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_clean_1 \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint=/path/to/checkpoint
# comment: test_clean_1 is an experimental split with 400K pairs.


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
    resume_from_checkpoint="/checkpoint/yaspar/13571189/last.ckpt"


# with MIMICIVCXR dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=biomedclip \
    experiment_name=biomedclip_retrieval_mimicivcxr \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.mimic=MIMICIVCXR \
    datasets.test.mimic.split=test \
    +datasets/transforms@datasets.test.mimic.transform=med_clip_vision_transform \
    datasets.test.mimic.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=0 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar/13571189/last.ckpt"
