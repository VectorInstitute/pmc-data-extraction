#!/bin/bash
#SBATCH -c 6
#SBATCH --partition=cpu
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=subcaption
#SBATCH --output=%x-%j.out

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

stdbuf -oL -eL srun python3 openpmcvl/pipeline/subcaption.py \
  --input-file /datasets/PMC-15M/0.jsonl \
  --output-file /datasets/PMC-15M/experimental/0.jsonl \
  --system-prompt-file openpmcvl/prompts/subcaption_system_prompt.txt \
  --base-url http://gpu010:8080/v1 \
  --model /model-weights/Meta-Llama-3.1-8B-Instruct \
  --max-tokens 500 \
  2>&1 | tee -a %x-%j.out