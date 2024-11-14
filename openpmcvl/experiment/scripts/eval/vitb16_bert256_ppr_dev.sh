# local train
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_train_1pair \
    dataloader.train.batch_size=32 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.text.clip_ckpt=null \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,100] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,100] \
    trainer.logger.wandb.offline=True

# local eval
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_eval_1pair \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.text.pretrained=True \
    task.encoders.text.clip_ckpt=null \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,100] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,100] \
    trainer.logger.wandb.offline=True
