#!/bin/bash
# Batch script to extract subfigures from compound figures using a detection model

#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=100GB
#SBATCH --time=15:00:00
#SBATCH --job-name=subfigure
#SBATCH --output=%x-%j.out

# Set environment variables:
# VENV_PATH: Path to virtual environment (e.g. export VENV_PATH=$HOME/venv)
# PROJECT_ROOT: Path to project root directory (e.g. export PROJECT_ROOT=$HOME/project)
# PMC_ROOT: Path to PMC dataset directory (e.g. export PMC_ROOT=$HOME/data)

# Sample command:
# sbatch openpmcvl/granular/pipeline/subfigure.sh 0 1 2 3 4 5 6 7 8 9 10 11

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
    # Define the paths for the evaluation file and the record file
    eval_file="$PMC_ROOT/${num}_meta.jsonl"
    rcd_file="$PMC_ROOT/${num}_subfigures.jsonl"

    # Run the subfigure separation script
    stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/subfigure.py \
      --separation_model openpmcvl/granular/checkpoints/subfigure_detector.pth \
      --eval_file "$eval_file" \
      --save_path "$PMC_ROOT/${num}_subfigures" \
      --rcd_file "$rcd_file" \
      --score_threshold 0.5 \
      --nms_threshold 0.4 \
      --batch_size 128 \
      --num_workers 8 \
      --gpu 0

    # Print a message indicating the completion of processing for the current JSONL number
    echo "Finished processing ${num}"
done
