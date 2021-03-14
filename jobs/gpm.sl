#!/bin/bash
#SBATCH --nodes 5
#SBATCH --ntasks-per-node 32
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00

#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -o gpm.out
#SBATCH -e gpm.err

conda activate myenv

srun --cpu_bind=cores python3 /global/cscratch1/sd/yzh/matrix/generate_pure_maps.py --nreal 1700 --outpath /global/cscratch1/sd/yzh/matrix/pure_maps


#cp <my_output_file> <target_location>/.
