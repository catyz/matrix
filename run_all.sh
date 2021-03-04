#!/bin/sh
python3 write_covariance.py
python3 write_eigs.py --n 1000
python3 write_purification.py --n 500
