#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --time=8:00:00
#SBATCH --job-name=classify
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/classify.py \
  --model_path openpmcvl/granular/models/resnext101_figure_class.pth \
  --dataset_path /datasets/PMC-15M/granular/subfigures.jsonl \
  --output_file /datasets/PMC-15M/granular/subfigures_classified.jsonl \
  --batch_size 128 \
  --num_workers 8 \
  --gpu 0
  2>&1 | tee -a %x-%j.out