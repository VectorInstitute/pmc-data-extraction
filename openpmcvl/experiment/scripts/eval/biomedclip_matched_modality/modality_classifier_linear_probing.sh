mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_classifier_linear_probing \
    experiment_name=5_modality_biomedclip \
    task.num_classes=5 \
    task.lr_scheduler.scheduler.T_max=2445 \
    job_type=train \
    ~task.postprocessors.norm \
    task.encoder.rgb.pretrained=True \
    task.postprocessors.logit_scale.logit_scale_init=4.4454 \
    task.postprocessors.logit_scale.learnable=True \
    task.encoder_checkpoint_path="/projects/DeepLesion/projects/pmc-data-extraction/outputs/biomedclip_matched_modality_linear_probing/2024-10-28/12-55-15/0_13808429/multimodal/bqe0tmpl/checkpoints/epoch\=39-step\=37080.ckpt" \









# ------------------------------ For BioMedCLIP -------------------------------------------
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_classifier_linear_probing \
    experiment_name=5_modality_biomedclip \
    task.num_classes=5 \
    task.lr_scheduler.scheduler.T_max=2445 \
    job_type=train \
    ~task.postprocessors.norm \
    task.encoder.rgb.pretrained=True \
    task.postprocessors.logit_scale.logit_scale_init=4.4454 \
    task.postprocessors.logit_scale.learnable=False







# ------------------------------ For Imagenet -------------------------------------------
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_linear_probing \
    experiment_name=5_modality_imagenet \
    task.num_classes=5 \
    task.lr_scheduler.scheduler.T_max=2445 \
    job_type=train \
    ~task.postprocessors.norm \
    task.encoder.rgb.pretrained=False \
    ~task.postprocessors.logit_scale