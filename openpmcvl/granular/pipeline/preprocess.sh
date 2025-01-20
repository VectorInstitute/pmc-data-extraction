#!/bin/bash
# Batch script to preprocess PMC figure-caption pairs

#SBATCH -c 32
#SBATCH --partition=cpu
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --job-name=preprocess
#SBATCH --output=%x-%j.out

# Set environment variables:
# VENV_PATH: Path to virtual environment (e.g. export VENV_PATH=$HOME/venv)
# PROJECT_ROOT: Path to project root directory (e.g. export PROJECT_ROOT=$HOME/project)
# PMC_ROOT: Path to PMC dataset directory (e.g. export PMC_ROOT=$HOME/data)

# Sample command:
# sbatch openpmcvl/granular/pipeline/preprocess.sh 0 1 2 3 4 5 6 7 8 9 10 11


# Activate virtual environment
source $VENV_PATH/bin/activate

# Set working directory
cd $PROJECT_ROOT

# Define the paths for the input and output files
INPUT_DIR="$PMC_ROOT"
OUTPUT_FILE="$PMC_ROOT/granular_meta.jsonl"
FIGURE_ROOT="$PMC_ROOT/figures"

# Check if the number of arguments is provided
if [ $# -eq 0 ]; then
    echo "Please provide JSONL numbers as arguments."
    exit 1
fi

# Get the list of JSONL numbers from the command line arguments
JSONL_NUMBERS="$@"

# Construct INPUT_FILES string
INPUT_FILES=""
for num in $JSONL_NUMBERS; do
    INPUT_FILES+="$INPUT_DIR/$num.jsonl "
done

# Run the preprocess script
stdbuf -oL -eL srun python3 openpmcvl/granular/pipeline/preprocess.py \
  --input_files $INPUT_FILES \
  --output_file $OUTPUT_FILE \
  --figure_root $FIGURE_ROOT \
  --keywords MRI fMRI CT CAT PET PET-MRI MEG EEG ultrasound X-ray Xray nuclear imaging tracer isotope scan positron EKG spectroscopy radiograph tomography endoscope endoscopy colonoscopy elastography ultrasonic ultrasonography echocardiogram endomicroscopy pancreatoscopy cholangioscopy enteroscopy retroscopy chromoendoscopy sigmoidoscopy cholangiography pancreatography cholangio-pancreatography esophagogastroduodenoscopy radiology pathology histopathology \
  2>&1 | tee -a %x-%j.out
