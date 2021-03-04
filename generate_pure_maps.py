import numpy as np
import healpy as hp
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate maps with pure E/B cls",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        '--nside',
        required=False,
        default=128,
        type=np.int,
        help='nside of map',
    )

    parser.add_argument(
        '--nreal',
        required=False,
        type=np.int,
        default=1,
        help='Number of maps'
    )
    
    parser.add_argument(
        '--beamfwhm',
        required=False,
        type=np.float,
        default=2.2,
        help='Beam fwhm in arcminutes'
    )

    args = parser.parse_args()
    return args

def prepend(cl):
    for c in cl:
        for i in range(2):
            c.insert(0,0)
    
def load_cl(args):
    lmax = 3*args.nside-1
    ell, TT, EE, BB, TE = np.loadtxt('totcls_lensed_r0p1.txt', unpack=True)
#     ell, TT, EE, BB, TE = np.loadtxt('test_dls.txt', unpack=True)

    #change the spectrum a little bit
#    EE *= 1.5
#    BB *=1.5

    total = np.array([ell, TT, EE, BB, TE])
    np.savetxt('test_dls.txt', np.transpose(total))

    prefactor = 2*np.pi/(ell * (ell + 1))
    prefactor[0] = 0    
    cl = np.array([TT, EE, BB, TE])
    cl *= prefactor
   
    return cl
    
def make_cl(args, mode):
    lmax = 3*args.nside-1
    
    if mode == 'E':
        cl_TT = np.zeros(lmax+1)
        cl_EE = np.ones(lmax+1)
        cl_BB = np.zeros(lmax+1)
        cl_TE = np.zeros(lmax+1)
        
    if mode == 'B':
        cl_TT = np.zeros(lmax+1)
        cl_EE = np.zeros(lmax+1)
        cl_BB = np.ones(lmax+1)
        cl_TE = np.zeros(lmax+1)
    
    cl_TT[0] = cl_TT[1] = 0
    cl_BB[0] = cl_BB[1] = 0
    cl_EE[0] = cl_EE[1] = 0
    cl_TE[0] = cl_TE[1] = 0
    
    
        
#     cl_TT = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax+1)]
#     #cl_EE = [0 for l in range(2, lmax)]
#     cl_BB = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax+1)]
#     cl_EE = [2*np.pi/(ell * (ell + 1)) for ell in range(2, lmax+1)]
#     #cl_BB = [0 for l in range(2, lmax)]
#     cl_TE = [0 for l in range(2, lmax+1)]

#    cl = cl_TT, cl_EE, cl_BB, cl_TE
#    prepend(cl)
    ell = np.arange(lmax+1)
    
    prefactor = 2*np.pi/(ell * (ell + 1))
    prefactor[0] = 0
    
    cl = np.array([cl_TT, cl_EE, cl_BB, cl_TE])
    cl *= prefactor
    
#     total = np.array([ell, cl_TT, cl_EE, cl_BB, cl_TE])
#     np.savetxt('test_dls.txt', np.transpose(total))
    
    return cl

def make_maps(args, cl, mode):
    for i in range(args.nreal):
        print(f'Processing {mode} realization {i}')
        m = hp.synfast(cl, args.nside, lmax=3*args.nside-1, pol=True, new=True)
        #m_smooth = hp.smoothing(m, args.beamfwhm *np.pi/10800)
        hp.write_map(f"./pure_maps/{mode}/{mode}_map_{i}.fits", hp.reorder(m, r2n=True), overwrite=True, nest= True, dtype=np.float64)

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
    make_E_maps(args)
    make_B_maps(args)
    
if __name__ == "__main__":
    main()