#!/bin/bash
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=align
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Set environment variables
# VENV_PATH: Path to your virtual environment (e.g. export VENV_PATH=$HOME/venv)
# PROJECT_ROOT: Path to project root directory (e.g. export PROJECT_ROOT=$HOME/project)
# PMC_ROOT: Path to PMC dataset directory (e.g. export PMC_ROOT=$HOME/data)

# Activate virtual environment 
source $VENV_PATH/bin/activate

# Set working directory
cd $PROJECT_ROOT

# Check if the correct number of arguments are provided
if [ $# -lt 2 ]; then
    echo "Please provide: begin index and end index as arguments."
    exit 1
fi

BEGIN_IDX=$1
END_IDX=$2

# Define the input and output files
input_file="$PMC_ROOT/pmc_oa.jsonl"
output_file="$PMC_ROOT/pmc_oa_aligned_${BEGIN_IDX}_${END_IDX}.jsonl"

# Print the alignment range
echo "Aligning from index ${BEGIN_IDX} to ${END_IDX}"

# Run the alignment script
stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/align.py \
  --root_dir "$PMC_ROOT" \
  --dataset_path "$input_file" \
  --save_path "$output_file" \
  --dataset_slice "${BEGIN_IDX}:${END_IDX}"

echo "Finished aligning from index ${BEGIN_IDX} to ${END_IDX}"