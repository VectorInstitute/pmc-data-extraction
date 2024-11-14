# train on rtx6000
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=long \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=32 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=2880 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_train_1pair_lr1e-5 \
    dataloader.train.batch_size=32 \
    dataloader.train.num_workers=4 \
    task.encoders.patient_q.pretrained=True \
    task.encoders.patient_t.pretrained=True \
    task.encoders.patient_q.clip_ckpt="" \
    task.encoders.patient_t.clip_ckpt="" \
    task.optimizer.lr=1e-5 \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar//last.ckpt"
# comment: test_clean_1 is an experimental split with 400K pairs.

# train on a40
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=a40_arashaf_multimodal \
    hydra.launcher.partition=a40 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=2880 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_train_1pair_lr1e-5 \
    dataloader.train.batch_size=128 \
    dataloader.train.num_workers=4 \
    task.encoders.patient_q.pretrained=True \
    task.encoders.patient_t.pretrained=True \
    task.encoders.patient_q.clip_ckpt="" \
    task.encoders.patient_t.clip_ckpt="" \
    task.optimizer.lr=1e-5 \
    trainer.callbacks.early_stopping.patience=1000 \
    strict_loading=False \
    resume_from_checkpoint="/projects/multimodal/checkpoints/openpmcvl/batch_size_tuning/bs_256/epoch\=18-step\=62149.ckpt"



# eval on rtx6000
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=8 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_eval \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.patient_q.pretrained=False \
    task.encoders.patient_t.pretrained=False \
    task.encoders.patient_q.clip_ckpt="" \
    task.encoders.patient_t.clip_ckpt="" \
    task.evaluation_tasks.retrieval.task.task_specs.0.top_k=[10,100,1000] \
    task.evaluation_tasks.retrieval.task.task_specs.1.top_k=[10,100,1000] \
    strict_loading=True \
    resume_from_checkpoint=""
