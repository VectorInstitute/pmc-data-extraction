#!/bin/bash
#SBATCH --job-name=pmc-subcaption-qwen32b
#SBATCH --partition=a100
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=59G
#SBATCH --output=qwen32b-subcap.%j.out

# Activate your environment

echo "Script Run Start!"
nvidia-smi

#module load cuda-12.4
module load gcc-12.3.0
gcc --version

source ~/envs/exp/bin/activate # Adjust this path to your virtual environment

echo "Module Loaded and Environment Activated!"

# Specify which GPUs to use
CUDA_VISIBLE_DEVICES=0,1 \
python /path/to/generate_subcaption_vllm.py \
  --data_path /path/to/data.csv \
  --model_dir /path/to/qwen2.5_vl_32B_model_weights_directory \
  --batch_size 32 \
  --max_new_tokens 1024 \
  --tp_size 2 \
  --gpu_mem_util 0.90 \
  --dtype bfloat16
