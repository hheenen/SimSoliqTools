#!/bin/bash
#SBATCH -J job
#SBATCH -p xeon16	### xeon8; xeon16; xeon24
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 01:00:00
#SBATCH -o slurm.%j.out
#SBATCH -e err

#cd $SLURM_SUBMIT_DIR
module purge
module load intel
export PATH=/home/cat/zwang/bin/cattheory/vasp/5.4.4:$PATH

mpirun vasp_std

