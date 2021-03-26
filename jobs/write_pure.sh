#!/bin/bash -l

#OpenMP settings:
export OMP_NUM_THREADS=8
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load openmpi

NAME='wpure'

N=1
TPN=1
NREAL=64

CPT=$((64/TPN))

cat << EOF > $NAME.sl 
#!/bin/bash
#SBATCH --nodes $N
#SBATCH --ntasks-per-node $TPN
#SBATCH --cpus-per-task $CPT
#SBATCH --time 00:30:00

#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -o $NAME.out
#SBATCH -e $NAME.err

conda activate myenv

srun --cpu_bind=cores python3 /global/cscratch1/sd/yzh/matrix/write_pure.py --n 500

#cp <my_output_file> <target_location>/.
EOF

sbatch $NAME.sl