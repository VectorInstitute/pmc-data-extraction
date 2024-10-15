# with OpenPMC-VL dataset
## local
mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    dataloader.test.batch_size=8 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    > outputs_bytes.txt
# comment: test_clean_1 is an experimental split with 400K pairs.

## on slurm
mmlearn_run --multirun hydra.launcher.mem_gb=64 \
    hydra.launcher.qos=normal \
    hydra.launcher.partition=rtx6000 \
    hydra.launcher.gres=gpu:1 \
    hydra.launcher.cpus_per_task=16 \
    hydra.launcher.tasks_per_node=1 \
    hydra.launcher.nodes=1 \
    hydra.launcher.stderr_to_stdout=true \
    hydra.launcher.timeout_min=900 \
    '+hydra.launcher.additional_parameters={export: ALL}' \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedclip_matched \
    experiment_name=biomedclip_retrieval_pmcvl \
    job_type=eval \
    dataloader.test.batch_size=64 \
    dataloader.test.num_workers=2 \
    task.encoders.text.pretrained=True \
    task.encoders.rgb.pretrained=True \
    datasets.train.pmcvl.split=train_dummy_ \
    datasets.val.pmcvl.split=test_dummy_ \
    datasets.test.pmcvl.split=test_dummy_ \
    trainer.logger.wandb.offline=True \
    > outputs_bytes.txt
