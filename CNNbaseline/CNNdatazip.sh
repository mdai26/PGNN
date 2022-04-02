#!/usr/bin/env zsh
#SBATCH -p research
#SBATCH -t 24:00:00 # time (D-HH:MM)
#SBATCH -n 1
#SBATCH -o CNN-%j.out
#SBATCH -e CNN-%j.err
#SBATCH --mem=16gb
#SBATCH --array=1-30%1

module load anaconda/mini/4.9.2
bootstrap_conda
conda activate cgcnn
python CNNdatazip.py --group $SLURM_ARRAY_TASK_ID
