#!/usr/bin/env zsh
#SBATCH -p research
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --array=0-99%1

module load intel/compiler
module load intel/mkl
module load fftw/impi/3.3.10

foldername="data_$SLURM_ARRAY_TASK_ID"
cd "$foldername"
mpirun -n 8 ./EffPropertyPoly.exe
cd ..
