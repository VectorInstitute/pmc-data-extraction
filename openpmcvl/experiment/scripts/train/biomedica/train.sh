#!/bin/bash

#SBATCH --job-name=pmcoa2_matched_train_ogtoken
#SBATCH --mem=0
#SBATCH --qos=a100_arashaf
#SBATCH --partition=a100
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --time=36:00:00
#SBATCH --export=ALL
#SBATCH --output=outputs/slurm-%j-%N.out
#SBATCH --open-mode=append

# load virtual environment
source /h/sajadra/dab-detr/venv/bin/activate

cd /h/sajadra/pmc-data-extraction
export PYTHONPATH="/h/sajadra/pmc-data-extraction"

export TORCH_NCCL_ASYNC_ERROR_HANDLING=3
export HYDRA_FULL_ERROR=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=45678

nvidia-smi
echo SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID}
echo SLURM_JOBID=${SLURM_JOBID}

# “srun” executes the script <ntasks-per-node * nodes> times
srun mmlearn_run \
    'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=biomedica_matched \
    experiment_name=pmcoa2_matched_train_ogtoken \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=512 \
    dataloader.train.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    trainer.max_epochs=64 \
    task.lr_scheduler.scheduler.t_max=24218 \
    task.lr_scheduler.scheduler.warmup_length=2421 \
    trainer.num_nodes=2 \
    trainer.devices=[0,1,2,3]
