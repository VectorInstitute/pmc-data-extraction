#!/bin/bash
#SBATCH --account=def-dolatab6
#SBATCH --job-name=openpmcvl
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --mem=2GB
#SBATCH --wait-all-nodes=1
#SBATCH --export=ALL
#SBATCH --output=outputs/slurm-%j-%N.out
#SBATCH --open-mode=append


# activate virtual environment
source /home/yaspar/envs/opmcvl/bin/activate

# print info
echo $(pwd)
echo $(date)
echo SLURM_NNODES=$SLURM_NNODES


# create the virtual environment on each node : 
srun --ntasks $SLURM_NNODES --tasks-per-node=1 python -u test.py

