#!/bin/bash -l

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

module load openmpi

NAME='wdata'

N=10
TPN=8
NREAL=1700

CPT=$((64/TPN))

cat << EOF > $NAME.sl 
#!/bin/bash
#SBATCH --nodes $N
#SBATCH --ntasks-per-node $TPN
#SBATCH --cpus-per-task $CPT
#SBATCH --time 00:15:00

#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -o $NAME.out
#SBATCH -e $NAME.err

conda activate myenv

srun --cpu_bind=cores python3 /global/cscratch1/sd/yzh/matrix/write_data.py --nreal $NREAL

#cp <my_output_file> <target_location>/.
EOF

sbatch $NAME.sl