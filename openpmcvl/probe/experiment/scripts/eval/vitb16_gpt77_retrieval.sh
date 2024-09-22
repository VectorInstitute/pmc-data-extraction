# with OpenPMC-VL dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=vitb16_gpt77 \
    experiment_name=vitb16_gpt77_retrieval \
    job_type=eval \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt


# with ROCO dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=vitb16_gpt77 \
    experiment_name=vitb16_gpt77_retrieval \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.roco=ROCO \
    datasets.test.roco.split=test \
    +datasets/transforms@datasets.test.roco.transform=med_clip_vision_transform \
    datasets.test.roco.transform.job_type=eval \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=HFCLIPTokenizer \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt


# with MIMICIVCXR dataset
mmlearn_run 'hydra.searchpath=[pkg://projects.openpmcvl.configs]' \
    +experiment=vitb16_gpt77 \
    experiment_name=vitb16_gpt77_retrieval \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.mimic=MIMICIVCXR \
    datasets.test.mimic.split=test \
    +datasets/transforms@datasets.test.mimic.transform=med_clip_vision_transform \
    datasets.test.mimic.transform.job_type=eval \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=HFCLIPTokenizer \
    dataloader.test.collate_fn.batch_processors.text.max_length=77 \
    dataloader.test.batch_size=32 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt
