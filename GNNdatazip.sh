#!/usr/bin/env zsh
#SBATCH -p research
#SBATCH -t 24:00:00 # time (D-HH:MM)
#SBATCH -n 1
#SBATCH -o GNNdata-%j.out
#SBATCH -e GNNdata-%j.err
#SBATCH --mem=16gb
#SBATCH --array=1-50%20

module load anaconda/mini/4.9.2
bootstrap_conda
conda activate cgcnn
python GNNdatazip.py --group $SLURM_ARRAY_TASK_ID
