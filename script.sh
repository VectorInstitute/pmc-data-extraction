#!/bin/bash
#SBATCH --job-name=pmcoa_training
#SBATCH --mem-per-gpu=64GB
#SBATCH --time=48:00:00
#SBATCH --nodes=8
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --output=outputs/slurm-%j-%N.out
#SBATCH --open-mode=append        

nvidia-smi

cd $SLURM_TMPDIR
mkdir work
cd work

tar -xf /home/neginb/projects/def-dolatab6/neginb/pmc_oa.tar

ls

export PMCOA_ROOT_DIR="$(pwd)/pmc_oa"

export HYDRA_FULL_ERROR=1
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT="multimodal"

cd /home/neginb/projects/def-dolatab6/neginb/pmc-data-extraction

mkdir outputs/$SLURM_JOB_ID

export PYTHONPATH="/home/neginb/projects/def-dolatab6/neginb/pmc-data-extraction"
module load gcc arrow/17.0.0
source env_2/bin/activate

srun mmlearn_run 'hydra.searchpath=[pkg://openpmcvl.experiment.configs]' \
    +experiment=pmcoa2_matched \
    experiment_name=pmcoa2_matched_train_ogtoken \
    datasets/tokenizers@dataloader.train.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.val.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    datasets/tokenizers@dataloader.test.collate_fn.batch_processors.text=BiomedCLIPTokenizerOG \
    dataloader.train.batch_size=4 \
    dataloader.val.batch_size=4 \
    dataloader.train.num_workers=4 \
    dataloader.val.num_workers=4 \
    task.encoders.text.pretrained=False \
    task.encoders.rgb.pretrained=False \
    task.lr_scheduler.scheduler.t_max=13636 \
    task.lr_scheduler.scheduler.warmup_length=2000 \
    ~trainer.callbacks.early_stopping

cd $SLURM_TMPDIR
tar -cf ~/projects/def-foo/johndoe/results.tar work

