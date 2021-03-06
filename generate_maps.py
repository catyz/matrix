import numpy as np
import healpy as hp
import argparse
import os
from mpi4py import MPI
import time

def add_map_args(parser):
    parser.add_argument(
        '--nside',
        required=False,
        default=64,
        type=int,
        help='nside of map',
    )
    parser.add_argument(
        '--pure',
        required=False,
        action='store_true',
        help='Flat spectrum or not'
    )

    parser.add_argument(
        '--nreal',
        required=False,
        type=int,
        default=1,
        help='Number of maps'
    )
    
    parser.add_argument(
        '--beamfwhm',
        required=False,
        type=float,
        default=2.2,
        help='Beam fwhm in arcminutes'
    )
    
    parser.add_argument(
        '--outpath',
        required=False,
        default='/global/cscratch1/sd/yzh/matrix/healpy_maps',
        help='Output directory',
    )
    
def make_cl(args, comm, mode):
    rank = comm.Get_rank()

    if rank == 0:
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

        ell = np.arange(lmax+1)

        #prefactor = 2*np.pi/(ell * (ell + 1))
        prefactor = 1/ell**2
        prefactor[0] = 0

        cl = np.array([cl_TT, cl_EE, cl_BB, cl_TE])
        cl *= prefactor

        #     total = np.array([ell, cl_TT, cl_EE, cl_BB, cl_TE])
        #     np.savetxt('test_dls.txt', np.transpose(total))
    else:
        cl = None
    
    cl = comm.bcast(cl, root=0)

    return cl

def load_cl(args, comm):
    rank = comm.Get_rank()
    
    if rank == 0:
        lmax = 3*args.nside-1
        ell, TT, EE, BB, TE = np.loadtxt('totcls_lensed_r0p1.txt', unpack=True)
        print('Loaded lensed r0p1')
    #     ell, TT, EE, BB, TE = np.loadtxt('test_dls.txt', unpack=True)

        #change the spectrum a little bit
    #    EE *= 0
    #    BB *=1.5

#         total = np.array([ell, TT, EE, BB, TE])
#         np.savetxt('test_dls.txt', np.transpose(total))

        prefactor = 2*np.pi/(ell * (ell + 1))
        prefactor[0] = 0    
        cl = np.array([TT, EE, BB, TE])
        cl *= prefactor
    else:
        cl = None
   
    cl = comm.bcast(cl, root=0)
    
    return cl

def make_maps(args, comm, mode):
    hp.disable_warnings()
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    sigmab = hp.nside2resol(args.nside)
    
    if rank == 0:
        reals = np.arange(args.nreal)
        chunks = np.array_split(reals, nprocs)
    else:
        chunks = None
        
    chunk = comm.scatter(chunks, root=0)
    
    if mode is not False:
        if mode == 'E':
            cl = make_cl(args, comm, mode)
        if mode == 'B':
            cl = make_cl(args, comm, mode)
    else:
        cl = load_cl(args, comm)
        
    for i in range(chunk[0], chunk[-1]+1):
        if mode is not False:
            print(f'Rank {rank} is processing {mode} realization {i} on processor {name}')
            outdir = f"{args.outpath}/{mode}"
        else:
            print(f'Rank {rank} is processing realization {i} on processor {name}')
            outdir = f"{args.outpath}"

        m = hp.synfast(cl, args.nside, sigma=sigmab, lmax=3*args.nside-1, pol=True, new=True)
        #m_smooth = hp.smoothing(m, args.beamfwhm *np.pi/10800)
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        hp.write_map(f"{outdir}/map_{i}.fits", hp.reorder(m, r2n=True), overwrite=True, nest= True, dtype=np.float64)

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    add_map_args(parser)
    args = parser.parse_args()
    
    comm = MPI.COMM_WORLD
    
    if args.pure is True:
        make_maps(args, comm, 'E')
        make_maps(args, comm, 'B')
    
    else:
        make_maps(args, comm, False)
    
    if comm.Get_rank() == 0:
        print(f'Completed in {time.time()-t0}')
        
if __name__ == "__main__":
    main()