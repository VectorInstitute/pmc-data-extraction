#!/bin/bash
# Batch script to align subfigures with subcaptions

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

# Sample command:
# sbatch openpmcvl/granular/pipeline/align.sh 0 1 2 3 4 5 6 7 8 9 10 11


# Activate virtual environment
source $VENV_PATH/bin/activate

# Set working directory
cd $PROJECT_ROOT

# Check if the number of arguments is provided
if [ $# -eq 0 ]; then
    echo "Please provide JSONL numbers as arguments."
    exit 1
fi

# Get the list of JSONL numbers from the command line arguments
JSONL_NUMBERS="$@"

# Iterate over each JSONL number
for num in $JSONL_NUMBERS; do
    input_file="$PMC_ROOT/${num}_subfigures_classified.jsonl"
    output_file="$PMC_ROOT/${num}_aligned.jsonl"

    # Run the alignment script
    stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/align.py \
        --root_dir "$PMC_ROOT" \
        --dataset_path "$input_file" \
        --save_path "$output_file"

    echo "Finished aligning ${num}"
done
