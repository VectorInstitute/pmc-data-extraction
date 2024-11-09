#!/bin/bash

#SBATCH --account=def-dolatab6
#SBATCH --job-name=PMCOA
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=10GB
#SBATCH --wait-all-nodes=1
#SBATCH --export=ALL
#SBATCH --output=outputs/pmcoa-slurm-%j-%N.out
#SBATCH --open-mode=append

export GPUS_PER_NODE=2

# Setting up dataset
cd $SLURM_TMPDIR
mkdir work
cd work
tar -xf /home/neginb/projects/def-dolatab6/neginb/pmc_oa.tar
rm $SLURM_TMPDIR/work/pmc_oa/images
ln -s $SLURM_TMPDIR/work/pmc_oa/ProcessedImages/caption_T060_filtered_top4_sep_v0_subfigures/ $SLURM_TMPDIR/work/pmc_oa/images
export PMCOA_ROOT_DIR="$(pwd)/pmc_oa"

# load virtual environment
export HYDRA_FULL_ERROR=1
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT="multimodal"
cd /home/neginb/projects/def-dolatab6/neginb/pmc-data-extraction
mkdir outputs/$SLURM_JOB_ID
export PYTHONPATH="/home/neginb/projects/def-dolatab6/neginb/pmc-data-extraction"
module load MistEnv/2021a cuda gcc anaconda3 cmake cudnn swig sox/14.4.2
module load gcc arrow/17.0.0
source env_2/bin/activate

echo $(pwd)
echo $(date)
echo SLURM_NNODES=$SLURM_NNODES


MASTER=$(/bin/hostname -s)
MPORT=$(shuf -i 6000-9999 -n 1)

# nvidia-smi
echo MASTER_ADDR=${MASTER}
echo MASTER_PORT=${MPORT}
echo SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}
echo SLURM_JOBID=${SLURM_JOBID}


srun --ntasks $SLURM_NNODES --tasks-per-node=1 mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched \
    experiment_name=pmcoa2_matched_train_ogtoken \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.val.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=32 \
    dataloader.val.batch_size=32 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.t_max=3409 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping \
    trainer.logger.wandb.offline=True \
    trainer.num_nodes=2
