#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --time 00:30:00

#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -o wpure.out
#SBATCH -e wpure.err

conda activate myenv

srun --cpu_bind=cores python3 /global/cscratch1/sd/yzh/matrix/write_pure.py --n 500

#cp <my_output_file> <target_location>/.
