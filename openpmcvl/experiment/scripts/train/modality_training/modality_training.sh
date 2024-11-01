# on slurm
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=a100_dolatae \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:4 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=4 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=4320 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched_modality \
    experiment_name=biomedclip_matched_train_ogtoken \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.val.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.t_max=104671 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping \
    resume_from_checkpoint="/projects/DeepLesion/projects/pmc-data-extraction/outputs/biomedclip_matched_modality_linear_probing/2024-10-28/12-55-15/0_13808429/multimodal/bqe0tmpl/checkpoints/epoch\=39-step\=37080.ckpt" \
    strict_loading=False \
    trainer.logger.wandb.id="" \
    trainer.logger.wandb.resume="must"
