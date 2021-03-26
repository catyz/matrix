#!/bin/sh
srun -n 8 python3 write_covariance.py -nreal 1700
python3 write_eigs.py -e 1000
python3 write_purification.py -n 500
