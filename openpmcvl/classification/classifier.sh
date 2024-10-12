#!/bin/bash
#SBATCH -c 6
#SBATCH --partition=t4v2
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --job-name=classify_images
#SBATCH --output=%x-%j.out
#SBATCH --gres=gpu:1

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

INPUT_FILE="openpmcvl/classification/embeddings.pt"
OUTPUT_FILE="openpmcvl/classification/classified.pt"

stdbuf -oL -eL srun python3 openpmcvl/classification/classifier.py \
  $INPUT_FILE \
  $OUTPUT_FILE \
  --classes "radiology" "pathology" "chest x-ray" \
  2>&1 | tee -a %x-%j.out
