mmlearn_run --multirun hydra.launcher.mem_gb=80 \
    hydra.launcher.qos=a100_dolatae \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched_retrieval \
    experiment_name=vitb16_bert256_zeroshot_bs256_matched_nw4 \
    dataloader.val.batch_size=32 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    job_type=eval \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch\=31-step\=104672.ckpt"



mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_mimic \
    job_type=eval \
    ~datasets.test.pmcvl \
    +datasets@datasets.test.mimic=MIMICIVCXR \
    datasets.test.mimic.split=test \
    +datasets/transforms@datasets.test.mimic.transform=biomedclip_vision_transform \
    datasets.test.mimic.transform.job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,50,200] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,50,200] \
    strict_loading=True \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch\=31-step\=104672.ckpt"
