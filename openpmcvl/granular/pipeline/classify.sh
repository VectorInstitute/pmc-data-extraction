#!/bin/bash
# Batch script to classify subfigures into figure types

#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=100GB
#SBATCH --time=15:00:00
#SBATCH --job-name=classify
#SBATCH --output=%x-%j.out

# Set environment variables:
# VENV_PATH: Path to virtual environment (e.g. export VENV_PATH=$HOME/venv)
# PROJECT_ROOT: Path to project root directory (e.g. export PROJECT_ROOT=$HOME/project)
# PMC_ROOT: Path to PMC dataset directory (e.g. export PMC_ROOT=$HOME/data)

# Sample command:
# sbatch openpmcvl/granular/pipeline/classify.sh 0 1 2 3 4 5 6 7 8 9 10 11


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
    input_file="$PMC_ROOT/${num}_subfigures.jsonl"
    output_file="$PMC_ROOT/${num}_subfigures_classified.jsonl"

    # Run the classification script
    stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/classify.py \
      --model_path openpmcvl/granular/checkpoints/resnext101_figure_class.pth \
      --dataset_path "$input_file" \
      --output_file "$output_file" \
      --batch_size 256 \
      --num_workers 8 \

    echo "Finished classifying ${num}"
done
