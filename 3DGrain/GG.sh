#!/bin/sh
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load gcc/9.4.0
GFORTRAN_UNBUFFERED_ALL='Y' 
time ./3dGG_OpenMP 8
