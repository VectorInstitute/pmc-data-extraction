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
    experiment_name=biomedclip_ppr_train_1pair \
    dataloader.train.batch_size=32 \
    dataloader.train.num_workers=4 \
    task.encoders.patient_q.pretrained=True \
    task.encoders.patient_t.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar//last.ckpt"
# comment: test_clean_1 is an experimental split with 400K pairs.

# eval on rtx6000
mmlearn_run --multirun hydra.launcher.mem_gb=0 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=32 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=600 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_ppr \
    experiment_name=biomedclip_ppr_eval_1pair \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=4 \
    task.encoders.patient_q.pretrained=True \
    task.encoders.patient_t.pretrained=True \
    strict_loading=False \
    resume_from_checkpoint="/checkpoint/yaspar//last.ckpt"
# comment: test_clean_1 is an experimental split with 400K pairs.
