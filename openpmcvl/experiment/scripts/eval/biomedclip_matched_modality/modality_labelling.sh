# ------------------- Using Imagenet checkpoint ------------------------------------------------------------

mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_labelling \
    experiment_name=5_modality_labelling \
    task.num_classes=5 \
    job_type=eval \
    task.output_file_name="pmcoa_2_test_imagenet_5_labels" \
    task.encoder_checkpoint_path="/projects/DeepLesion/projects/pmc-data-extraction/outputs/5_modality_imagenet/2024-11-01/23-07-49/0_13825588/multimodal/if9l9snq/checkpoints/epoch\=39-step\=2480.ckpt"

