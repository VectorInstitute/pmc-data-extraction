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
    resume_from_checkpoint=""
# comment: test_clean_1 is an experimental split with 400K pairs.


# with OpenPMC-VL
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    dataloader.test.batch_size=128 \
    dataloader.test.num_workers=4 \
    strict_loading=False \
    resume_from_checkpoint=path/to/checkpoint
