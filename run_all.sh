#!/bin/sh
# srun -n 8 python write_data.py --nreal 4000 --mask south_patch_apo_64.fits
# srun -n 2 python write_cov.py
# python write_eigs.py -e 2000
python write_pure.py -e 2000 -n 2000 --mask south_patch_apo_64.fits

# srun -n 4 python generate_maps.py --nreal 100
# python namaster.py