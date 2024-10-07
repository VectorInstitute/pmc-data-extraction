#!/bin/bash
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=subfigure
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

stdbuf -oL -eL srun python3 openpmcvl/pipeline/align.py \
  --dataset_path /datasets/PMC-15M/experimental/demo/demo_subfigures.jsonl \
  --save_path /datasets/PMC-15M/experimental/demo/demo_subfigures_labeled.jsonl \
  2>&1 | tee -a %x-%j.out