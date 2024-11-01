mmlearn_run --multirun hydra.launcher.mem_gb=80 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_ZSC \
    experiment_name=biomedclip_matched_modality_ZSC \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    strict_loading=True \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch\=31-step\=104672.ckpt"
