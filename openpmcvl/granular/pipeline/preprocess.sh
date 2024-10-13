#!/bin/bash
#SBATCH -c 8
#SBATCH --partition=cpu
#SBATCH --mem=48GB
#SBATCH --time=12:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=%x-%j.out

source /h/afallah/light/bin/activate

cd /h/afallah/pmc-data-extraction

INPUT_DIR="/datasets/PMC-15M"
OUTPUT_FILE="/datasets/PMC-15M/granular/granular_meta.jsonl"
FIGURE_ROOT="/datasets/PMC-15M/figures"

# Specify which JSONL files to process
JSONL_NUMBERS="0 1 2 3 4 5 6 7 8 9 10 11"

# Construct INPUT_FILES string
INPUT_FILES=""
for num in $JSONL_NUMBERS; do
    INPUT_FILES+="$INPUT_DIR/$num.jsonl "
done

stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/preprocess.py \
  --input_files $INPUT_FILES \
  --output_file $OUTPUT_FILE \
  --figure_root $FIGURE_ROOT \
  --keywords MRI fMRI CT CAT PET PET-MRI MEG EEG ultrasound X-ray Xray nuclear imaging tracer isotope scan positron EKG spectroscopy radiograph tomography endoscope endoscopy colonoscopy elastography ultrasonic ultrasonography echocardiogram endomicroscopy pancreatoscopy cholangioscopy enteroscopy retroscopy chromoendoscopy sigmoidoscopy cholangiography pancreatography cholangio-pancreatography esophagogastroduodenoscopy radiology pathology histopathology \
  2>&1 | tee -a %x-%j.out