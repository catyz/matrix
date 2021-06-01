#!/bin/sh
mpirun -n 8 python write_data.py --nreal 4096 --mask south_patch_apo_64.fits
mpirun -n 2 python write_cov.py
python write_eigs.py -e 3072
#python write_pure.py -e 3072 -c 1.02 --mask south_patch_apo_64.fits

# mpirun -n 4 python generate_maps.py --nreal 100
# python namaster.py