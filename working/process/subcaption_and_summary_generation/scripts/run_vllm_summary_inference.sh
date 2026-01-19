#!/bin/bash
#SBATCH --job-name=summary-pmc
#SBATCH --partition=a40
#SBATCH --qos=scavenger
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=43G
#SBATCH --output=qwen14b-summary.%j.out

echo "Script Run Start!"
nvidia-smi

#module load cuda-12.4
module load gcc-12.3.0
gcc --version

source ~/envs/exp2/bin/activate # Adjust this path to your virtual environment

echo "Module Loaded and Environment Activated!"

# Specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1 \
python /path/to/generate_summary_vllm.py \
  --data_path /path/to/data.csv \
  --model_dir /path/to/qwen2.5_14b_instruct_model_weights \
  --batch_size 1024 \
  --max_new_tokens 256 \
  --tp_size 2 \
  --gpu_mem_util 0.90 \
  --dtype bfloat16

