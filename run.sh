#!/usr/bin/env zsh
#SBATCH -p research
#SBATCH -t 02:00:00 # time (D-HH:MM)
#SBATCH -G 1
#SBATCH -o run-%j.out
#SBATCH -e run-%j.err

module load anaconda/mini/4.9.2
bootstrap_conda
conda activate cgcnn
python main.py
