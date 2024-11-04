# ----------------------- Our PMCOA checkpoint --------------------------
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
    +experiment=med_benchmarking_ZSC \
    experiment_name=med_benchmarking_ZSC_bs128_pmcoa_ep31 \
    dataloader.val.batch_size=32 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    strict_loading=True \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/pmc_oa_2/bs_128/13831347/epoch\=31-step\=27296.ckpt"
    
    

# ------------------------------------- BIOMEDCLIP correct logit scale ------------------------------------
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
    +experiment=med_benchmarking_ZSC \
    experiment_name=med_benchmarking_ZSC_biomedclip_with_correct_logitscale \
    dataloader.val.batch_size=32 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    ~task.postprocessors.norm_and_logit_scale.norm \
    task.postprocessors.norm_and_logit_scale.logit_scale.logit_scale_init=4.4454 \
    task.postprocessors.norm_and_logit_scale.logit_scale.learnable=False
    
    "/projects/multimodal/checkpoints/mmlearn/med_benchmarking/vit_base_patch16_224_ep11.ckpt"