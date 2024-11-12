# train on a40 biomedclip
mmlearn_run --multirun hydra.launcher.mem_gb=128 \
    hydra.launcher.account=deadline \
    hydra.launcher.qos=deadline \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_train \
    dataloader.train.batch_size=256 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.text.clip_ckpt="/projects/multimodal/checkpoints/openpmcvl/pmc_oa_2/bs_256/epoch\=31-step\=13664.ckpt"  \
    task.lr_scheduler.scheduler.t_max=5027 \
    task.lr_scheduler.scheduler.warmup_length=126

# eval on rtx6000 biomedclip
mmlearn_run --multirun hydra.launcher.mem_gb=128 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:2 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=2 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_eval \
    job_type=eval \
    dataloader.test.batch_size=1024 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.text.clip_ckpt="/projects/multimodal/checkpoints/openpmcvl/pmc_oa_2/bs_256/epoch\=31-step\=13664.ckpt" \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[1000] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[1000] \
    strict_loading=True \
    resume_from_checkpoint=""
