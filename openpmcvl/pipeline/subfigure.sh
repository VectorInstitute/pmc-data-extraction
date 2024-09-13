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

stdbuf -oL -eL srun python3 openpmcvl/pipeline/subfigure.py \
  --input-file /datasets/PMC-15M/experimental/0.jsonl \
  --output-file /datasets/PMC-15M/experimental/0.jsonl \
  --data-root /datasets/PMC-15M \
  --model-path openpmcvl/models/subfigure_detector.pth \
  --detection-threshold 0.6 \
  2>&1 | tee -a %x-%j.out