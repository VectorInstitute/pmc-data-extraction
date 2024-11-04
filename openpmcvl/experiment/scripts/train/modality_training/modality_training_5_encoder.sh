# on slurm
mmlearn_run --multirun hydra.launcher.mem_gb=200 \
    hydra.launcher.qos=a100_dolatae \
    hydra.launcher.partition=a100 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=3000 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_specific_5_encoder \
    experiment_name=modality_specific_5_encoder \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.val.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=256 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.d3.pretrained=False \
    task.encoders.dm.pretrained=False \
    task.encoders.ds.pretrained=False \
    task.encoders.dr.pretrained=False \
    task.encoders.dv.pretrained=False \
    task.lr_scheduler.scheduler.t_max=13636 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping \
    strict_loading=False \
    resume_from_checkpoint="path/to/checkpoint" \
    trainer.logger.wandb.id="" \
    trainer.logger.wandb.resume="must"









# ---------------------------------------------------------- A40 -----------------------------------------------------
mmlearn_run --multirun hydra.launcher.mem_gb=100 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=4 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=3000 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=modality_specific_5_encoder \
    experiment_name=modality_specific_5_encoder \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.val.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=64 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.d3.pretrained=False \
    task.encoders.dm.pretrained=False \
    task.encoders.ds.pretrained=False \
    task.encoders.dr.pretrained=False \
    task.encoders.dv.pretrained=False \
    task.lr_scheduler.scheduler.t_max=54544 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping