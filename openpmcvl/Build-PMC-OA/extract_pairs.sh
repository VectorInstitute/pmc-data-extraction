#!/bin/bash
#SBATCH --job-name=pmc15m
#SBATCH --ntasks=1
#SBATCH -c 8
#SBATCH --mem=50GB
#SBATCH --time=4-00:00:00  
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --export=ALL
#SBATCH --output=outputs/%x.%j.log  


# determine which volume to parse
export VOL=$1

echo $(date)
echo hostname = $(hostname)
echo VOLUME NUMBER = ${VOL}

# # load anaconda
# module use /pkgs/environment-modules/
# module load anaconda/3.9

# activate virtual env
source ~/.bashrc
conda activate ~/Documents/envs/pubmed/

# move to correct dir
cd ~/Documents/GitHub/Multimodal/pmc_dataset/Build-PMC-OA

# run program
python src/fetch_oa.py --extraction-dir /datasets/PMC-15M --volumes ${VOL} --no-download
