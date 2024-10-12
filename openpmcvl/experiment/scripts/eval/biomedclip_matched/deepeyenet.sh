# with DeepEyeNet dataset
mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_deepeyenet \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.dey=DeepEyeNet \
    datasets.test.dey.split=test \
    +datasets/transforms@datasets.test.dey.transform=biomedclip_vision_transform \
    datasets.test.dey.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=256 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=False \
    resume_from_checkpoint=""