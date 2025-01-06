#!/bin/bash
#SBATCH -c 12
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=100GB
#SBATCH --time=15:00:00
#SBATCH --job-name=classify
#SBATCH --output=%x-%j.out

# Activate the environment
source /h/afallah/light/bin/activate

# Set the working directory
cd /h/afallah/pmc-data-extraction

# Check if the number of arguments is provided
if [ $# -eq 0 ]; then
    echo "Please provide JSONL numbers as arguments."
    exit 1
fi

# Get the list of JSONL numbers from the command line arguments
JSONL_NUMBERS="$@"

# Iterate over each JSONL number
for num in $JSONL_NUMBERS; do
    input_file="/datasets/PMC-15M/granular/${num}_subfigures.jsonl"
    output_file="/datasets/PMC-15M/granular/${num}_subfigures_classified.jsonl"

    # Run the classification script
    stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/classify.py \
      --model_path openpmcvl/granular/models/resnext101_figure_class.pth \
      --dataset_path "$input_file" \
      --output_file "$output_file" \
      --batch_size 256 \
      --num_workers 8 \
    
    echo "Finished classifying ${num}"
done