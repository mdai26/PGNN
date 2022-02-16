#!/bin/sh
#SBATCH -p RM-shared
#SBATCH -t 08:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node 8
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

module load intel/20.4
module load intelmpi/20.4-intel20.4
mpirun -n 8 ./EffPropertyPoly.exe
