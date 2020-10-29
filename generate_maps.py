import numpy as np
import healpy as hp
import argparse

parser = argparse.ArgumentParser(
        description="Generate 1/l^2 spectrum E/B maps for matrix purification study",
        fromfile_prefix_chars="@",
    )

parser.add_argument(
    '--nside',
    required=False,
    default=512,
    help='nside of map',
)

parser.add_argument(
    '--mode',
    required=True,
    help='E or B'
)

parser.add_argument(
    '--nreal',
    required=False,
    default=1,
    help='Number of maps'
)

parser.add_argument(
    '--lmax',
    required=False,
    default=100,
    help='ell max'
)

parser.add_argument(
    '--outpath',
    required=True,
    help='out path'
)

args = parser.parse_args()

cl_TT = [0 for l in range(2, args.lmax)]
if args.mode == 'B':
    cl_EE = [0 for l in range(2, args.lmax)]
    cl_BB = [1/l**2 for l in range(2, args.lmax)]
if args.mode == 'E':
    cl_EE = [1/l**2 for l in range(2, args.lmax)]
    cl_BB = [0 for l in range(2, args.lmax)]
cl_TE = [0 for l in range(2, args.lmax)]
cl = cl_TT, cl_EE, cl_BB, cl_TE
for c in cl:
    for i in range(2):
        c.insert(0,0)
        
for i in range(args.nreal):
    print(f'Processing realization {i}')
    alms = hp.synalm(cl, new=True)
    m = hp.alm2map(alms, args.nside, pol=True)
    hp.write_map(f"{args.outpath}/{args.mode}_map_{i}.fits", m, overwrite=True, dtype=np.float32)
