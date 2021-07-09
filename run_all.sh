#!/bin/sh
# srun -n 8 python write_data.py --nreal 4096 --mask south_patch_apo_64.fits
# srun -n 2 python write_cov.py
python write_eigs.py -e 2500
python write_pure.py -e 2500 -c 1.02 --mask south_patch_apo_64.fits

srun -n 4 python generate_maps.py --nreal 100 --nside 64 
srun -n 4 python generate_maps.py --nreal 100 --nside 64 --pure

# python namaster.py

