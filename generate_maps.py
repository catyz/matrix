import numpy as np
import healpy as hp
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate 1/l^2 spectrum E/B maps for matrix purification study",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        '--nside',
        required=False,
        default=256,
        type=np.int,
        help='nside of map',
    )

    parser.add_argument(
        '--mode',
        required=False,
        default='both',
        type=np.str,
        help='E or B or both'
    )

    parser.add_argument(
        '--nreal',
        required=False,
        type=np.int8,
        default=1,
        help='Number of maps'
    )

    args = parser.parse_args()
    return args

def make_cl(args, mode):
    lmax = 3*args.nside-1
    cl_TT = [1/l**2 for l in range(2, lmax)]
    if mode == 'B':
        cl_EE = [0 for l in range(2, lmax)]
        cl_BB = [1/l**2 for l in range(2, lmax)]
    if mode == 'E':
        cl_EE = [1/l**2 for l in range(2, lmax)]
        cl_BB = [0 for l in range(2, lmax)]
    cl_TE = [0 for l in range(2, lmax)]
    cl = cl_TT, cl_EE, cl_BB, cl_TE
    for c in cl:
        for i in range(2):
            c.insert(0,0)
    return cl

def make_maps(args, cl, mode):
    for i in range(args.nreal):
        print(f'Processing {mode}, realization {i}')
        m = hp.synfast(cl, args.nside, pol=True, new=True)
        hp.write_map(f"./input_maps/{mode}/{mode}_map_{i}.fits", hp.reorder(m, r2n=True), overwrite=True, nest= True, dtype=np.float64)

def make_E_maps(args):
    mode = 'E'
    cl = make_cl(args, mode)
    make_maps(args, cl, mode)
        
def make_B_maps(args):
    mode = 'B'
    cl = make_cl(args, mode)
    make_maps(args, cl, mode)
    
def main():
    
    args=parse_arguments()
    
    if args.mode=='E':
        make_E_maps(args)
    if args.mode=='B':
        make_B_maps(args)
    if args.mode=='both':
        make_E_maps(args)
        make_B_maps(args)
    
if __name__ == "__main__":
    main()