#!/bin/bash
#SBATCH -c 6
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --job-name=align
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Activate the environment
source /h/afallah/light/bin/activate

# Set the working directory
cd /h/afallah/pmc-data-extraction

# Check if the correct number of arguments are provided
if [ $# -lt 2 ]; then
    echo "Please provide: begin index and end index as arguments."
    exit 1
fi

BEGIN_IDX=$1
END_IDX=$2

# Define the root directory
root_dir="/projects/multimodal/datasets/pmc_oa"

# Define the input and output files
input_file="${root_dir}/pmc_oa.jsonl"
output_file="${root_dir}/pmc_oa_labeled/pmc_oa_aligned_${BEGIN_IDX}_${END_IDX}.jsonl"

# Print the alignment range
echo "Aligning from index ${BEGIN_IDX} to ${END_IDX}"

# Run the alignment script
stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/align.py \
  --root_dir "$root_dir" \
  --dataset_path "$input_file" \
  --save_path "$output_file" \
  --dataset_slice "${BEGIN_IDX}:${END_IDX}"

echo "Finished aligning from index ${BEGIN_IDX} to ${END_IDX}"

# Original loop commented out:
# if [ $# -eq 0 ]; then
#     echo "Please provide JSONL numbers as arguments."
#     exit 1
# fi
# 
# JSONL_NUMBERS="$@"
# 
# for num in $JSONL_NUMBERS; do
#     input_file="/datasets/PMC-15M/granular/${num}_subfigures_classified.jsonl"
#     output_file="/datasets/PMC-15M/granular/${num}_subfigures_aligned.jsonl"
#     
#     stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/align.py \
#       --dataset_path "$input_file" \
#       --save_path "$output_file" \
#     
#     echo "Finished aligning ${num}"
# done