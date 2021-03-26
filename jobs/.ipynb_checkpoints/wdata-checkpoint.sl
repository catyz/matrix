#!/bin/bash
#SBATCH --nodes 10
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00

#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -o wdata.out
#SBATCH -e wdata.err

conda activate myenv

srun --cpu_bind=cores python3 /global/cscratch1/sd/yzh/matrix/write_data.py --nreal 1700

#cp <my_output_file> <target_location>/.
