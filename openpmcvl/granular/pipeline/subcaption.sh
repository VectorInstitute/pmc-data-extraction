#!/bin/bash
# Batch script to extract subcaptions from figure captions using GPT API

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

# Sample command:
# sbatch openpmcvl/granular/pipeline/subcaption.sh 0 1 2 3 4 5 6 7 8 9 10 11

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
    # Run the subcaption script
    stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/subcaption.py \
      --input-file "$PMC_ROOT/${num}_meta.jsonl" \
      --output-file "$PMC_ROOT/${num}_subcaptions.jsonl" \
      --max-tokens 500 \
      2>&1 | tee -a %x-%j.out

    echo "Finished processing ${num}"
done
