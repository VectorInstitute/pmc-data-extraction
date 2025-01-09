#!/bin/bash
#SBATCH -c 6
#SBATCH --partition=cpu
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=subcaption
#SBATCH --output=%x-%j.out

# Set environment variables:
# VENV_PATH: Path to virtual environment (e.g. export VENV_PATH=$HOME/venv)
# PROJECT_ROOT: Path to project root directory (e.g. export PROJECT_ROOT=$HOME/project)
# PMC_ROOT: Path to PMC dataset directory (e.g. export PMC_ROOT=$HOME/data)

# Activate virtual environment
source $VENV_PATH/bin/activate

# Set working directory
cd $PROJECT_ROOT

# Run the subcaption script
stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/subcaption.py \
  --input-file $PMC_ROOT/pmc_oa.jsonl \
  --output-file $PMC_ROOT/pmc_oa_caption.jsonl \
  --max-tokens 500 \
  2>&1 | tee -a %x-%j.out