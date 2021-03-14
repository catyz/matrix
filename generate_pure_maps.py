import numpy as np
import healpy as hp
import argparse
import os
from mpi4py import MPI
import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate maps with pure E/B cls",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        '--nside',
        required=False,
        default=128,
        type=int,
        help='nside of map',
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
        default='./pure_maps',
        help='Output directory',
    )

    args = parser.parse_args()
    return args

    
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

        prefactor = 2*np.pi/(ell * (ell + 1))
        prefactor[0] = 0

        cl = np.array([cl_TT, cl_EE, cl_BB, cl_TE])
        cl *= prefactor

        #     total = np.array([ell, cl_TT, cl_EE, cl_BB, cl_TE])
        #     np.savetxt('test_dls.txt', np.transpose(total))
    else:
        cl = None
    
    cl = comm.bcast(cl, root=0)

    return cl

def make_maps(args, comm, mode):
    
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    
    if rank == 0:
        reals = np.arange(args.nreal)
        chunks = np.array_split(reals, nprocs)
    else:
        chunks = None
        
    chunk = comm.scatter(chunks, root=0)
    
    if mode == 'E':
        cl = make_cl(args, comm, mode)
        
    if mode == 'B':
        cl = make_cl(args, comm, mode)
        
    for i in range(chunk[0], chunk[-1]+1):
        print(f'Rank {rank} is processing {mode} realization {i} on processor {name}')
        m = hp.synfast(cl, args.nside, lmax=3*args.nside-1, pol=True, new=True)
        #m_smooth = hp.smoothing(m, args.beamfwhm *np.pi/10800)
        outdir = f"{args.outpath}/{mode}"
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
        hp.write_map(f"{outdir}/{mode}_map_{i}.fits", hp.reorder(m, r2n=True), overwrite=True, nest= True, dtype=np.float64)

def main():
    t0 = time.time()
    args=parse_arguments()
    
    comm = MPI.COMM_WORLD
    make_maps(args, comm, mode='E')
    make_maps(args, comm, mode='B')
    
    if comm.Get_rank() == 0:
        print(f'Completed in {time.time()-t0}')
        
if __name__ == "__main__":
    main()