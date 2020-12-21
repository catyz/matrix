import numpy as np
import healpy as hp
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate maps with CAMB cls",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        '--nside',
        required=False,
        default=128,
        type=np.int,
        help='nside of map',
    )

#     parser.add_argument(
#         '--mode',
#         required=False,
#         default='both',
#         type=np.str,
#         help='E or B or both'
#     )

    parser.add_argument(
        '--nreal',
        required=False,
        type=np.int,
        default=1,
        help='Number of maps'
    )

    args = parser.parse_args()
    return args

def prepend(cl):
    for c in cl:
        for i in range(2):
            c.insert(0,0)
    
def load_cl(args):
    lmax = 3*args.nside-1
    total = np.loadtxt('totcls.txt')
    ell, TT, EE, BB, TE = np.transpose(total[0:lmax])
    TT = list(2*np.pi/(ell * (ell + 1))*TT)
    EE = list(2*np.pi/(ell * (ell + 1))*EE)
    BB = list(2*np.pi/(ell * (ell + 1))*BB)
    TE = list(2*np.pi/(ell * (ell + 1))*TE)
    cl = TT, EE, BB, TE
    prepend(cl)
    
    return cl
    
def make_cl(args, mode):
    lmax = 3*args.nside-1
    
    cl_TT = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax)]
    if mode == 'B':
        cl_EE = [0 for l in range(2, lmax)]
        cl_BB = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax)]
    if mode == 'E':
        cl_EE = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax)]
        cl_BB = [0 for l in range(2, lmax)]
    cl_TE = [0 for l in range(2, lmax)]
    cl = cl_TT, cl_EE, cl_BB, cl_TE
    prepend(cl)

    return cl

def make_maps(args):
    cl = load_cl(args)
    for i in range(args.nreal):
        print(f'Processing realization {i}')
        m = hp.synfast(cl, args.nside, pol=True, new=True)
        hp.write_map(f"./input_maps/map_{i}.fits", hp.reorder(m, r2n=True), overwrite=True, nest= True, dtype=np.float64)
        #hp.write_map(f"./input_maps/{mode}_Q/{mode}_Q_map_{i}.fits", hp.reorder(m[1], r2n=True), overwrite=True, nest= True, dtype=np.float64)
        #hp.write_map(f"./input_maps/{mode}_U/{mode}_U_map_{i}.fits", hp.reorder(m[2], r2n=True), overwrite=True, nest= True, dtype=np.float64)

# def make_E_maps(args):
#     mode = 'E'
#     cl = make_cl(args, mode)
#     make_maps(args, cl, mode)
        
# def make_B_maps(args):
#     mode = 'B'
#     cl = make_cl(args, mode)
#     make_maps(args, cl, mode)
    
def main():
    args=parse_arguments()
    make_maps(args)
    
if __name__ == "__main__":
    main()