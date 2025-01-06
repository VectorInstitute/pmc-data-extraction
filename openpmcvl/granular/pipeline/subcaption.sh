#!/bin/bash
#SBATCH -c 6
#SBATCH --partition=cpu
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=subcaption
#SBATCH --output=%x-%j.out

# Activate the environment
source /h/afallah/light/bin/activate

# Set the working directory
cd /h/afallah/pmc-data-extraction

# Run the subcaption script
stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/subcaption.py \
  --input-file /datasets/PMC-15M/experimental/demo/demo.jsonl \
  --output-file /datasets/PMC-15M/experimental/demo/demo_caption.jsonl \
  --base-url http://gpu030:8080/v1 \
  --model /model-weights/Meta-Llama-3.1-8B-Instruct \
  --max-tokens 500 \
  2>&1 | tee -a %x-%j.out