#!/usr/bin/env zsh
#SBATCH -p research
#SBATCH -t 2-00:00 # time (D-HH:MM)
#SBATCH -N 1
#SBTACH -n 1
#SBATCH -o data-%j.out
#SBATCH -e data-%j.err

module load anaconda/mini/4.9.2
bootstrap_conda
conda activate cgcnn
time python -W ignore datageneration.py
