#!/bin/bash
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=subfigure
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/subfigure.py \
  --separation_model openpmcvl/granular/models/subfigure_detector.pth \
  --class_model openpmcvl/granular/models/resnext101_figure_class.pth \
  --eval_file /datasets/PMC-15M/granular/granular_meta.jsonl \
  --img_root /datasets/PMC-15M/figures \
  --save_path /datasets/PMC-15M/granular/subfigures \
  --rcd_file /datasets/PMC-15M/granular/subfigures.jsonl  \
  --score_threshold 0.5 \
  --nms_threshold 0.4 \
  --batch_size 8 \
  --num_workers 4 \
  --gpu 0
  2>&1 | tee -a %x-%j.out