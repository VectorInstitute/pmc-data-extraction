# on rtx6000
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=long \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=32 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=2880 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_matched_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_clean \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch=18-step=62149.ckpt"
# comment: test_clean_1 is an experimental split with 400K pairs.

# on a40 with OpenPMC-VL test split
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_matched_eval_pmcvl \
    job_type=eval \
    datasets.test.pmcvl.split=test_clean \
    dataloader.test.batch_size=128 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch=18-step=62149.ckpt"

# on a40 with MIMIC-IV-CXR
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_matched_eval_mimic \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.mimic=MIMICIVCXR \
    datasets.test.mimic.split=test \
    +datasets/transforms@datasets.test.mimic.transform=biomedclip_vision_transform \
    datasets.test.mimic.transform.job_type=eval \
    dataloader.test.collate_fn.batch_processors.text.max_length=256 \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizer \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch18-step62149.ckpt"
