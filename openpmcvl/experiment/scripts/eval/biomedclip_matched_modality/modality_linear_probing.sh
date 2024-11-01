mmlearn_run --multirun hydra.launcher.mem_gb=64 \
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
    +experiment=modality_linear_probing \
    experiment_name=biomedclip_matched_modality_linear_probing \
    task.num_classes=3 \
    task.lr_scheduler.scheduler.T_max=2480 \
    job_type=eval \
    ~task.postprocessors.norm_and_logit_scale.norm \
    task.encoder.rgb.pretrained=True \
    task.postprocessors.logit_scale.logit_scale_init=4.4454 \
    task.postprocessors.logit_scale.learnable=False \
    task.encoder_checkpoint_path="/projects/DeepLesion/projects/pmc-data-extraction/outputs/biomedclip_matched_modality_linear_probing/2024-10-28/12-55-15/0_13808429/multimodal/bqe0tmpl/checkpoints/epoch\=39-step\=37080.ckpt" \

