import numpy as np
import healpy as hp
import scipy.sparse as sparse
import pymaster as nmt
from mpi4py import MPI
import argparse
import time
from glob import glob
import os
import functools
from generate_maps import make_cl

def add_cov_args(parser):
    parser.add_argument(
        '--nreal',
        required=False,
        default=1,
        type=int,
        help='Number of sims'
    )
    
    parser.add_argument(
        '--nside',
        required=False,
        default=128,
        type=int,
        help='Nside'
    )
    
    parser.add_argument(
        '--map-disk',
        required=False,
        action='store_true',
        help='Read maps from disk'
    )
    
    parser.add_argument(
        '--workdir',
        required=False,
        default='/global/cscratch1/sd/yzh/matrix',
        help='omegalul'
    )

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        hours = divmod(run_time,3600)
        minutes = divmod(hours[1],60)
        seconds = minutes[1]
        print(f'Finished {func.__name__!r} in {int(hours[0]):02d}:{int(minutes[0]):02d}:{seconds:04.1f}')
        return value
    return wrapper_timer

#@timer
def sparse_covariance(X):
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
        
    n = X.shape[0]
    means = np.array(X.mean(axis=0))[0]
    nz = np.where(means!=0)[0]
    
    #Center the data
    for i in nz:
        X[:,i] -= np.ones((n,1))*means[i]

    X = X.transpose()
    C = X.dot(X.transpose())/n
    
    return C.tocsr()
    
def remove_nans(m_vector):
    nan_indx = np.where(np.isnan(m_vector))[0]
    if len(nan_indx) !=0:
        #print(f'nans at {nan_indx}')
        m_vector[nan_indx] = 0
        
#@timer
def make_cov(args, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    
    nreal = args.nreal
    nside = args.nside
    npix = hp.nside2npix(nside)
    
    m_vector = np.zeros(2*npix)
    X_E = sparse.coo_matrix(m_vector)
    X_B = sparse.coo_matrix(m_vector)
    
    if rank == 0:
        #mask = hp.read_map(f'{args.workdir}/small_patch_apo.fits',verbose=False, dtype=np.float64)       
        mask = hp.read_map(f'{args.workdir}/south_patch_apo.fits', verbose=False, dtype=np.float64)        
        reals = np.arange(nreal)
        chunks = np.array_split(reals, size)
    else:
        chunks = None
        mask = None
    
    mask = comm.bcast(mask, root=0)
    chunk = comm.scatter(chunks, root=0)
    
    for i in chunk:
        print(f'Rank {rank} processing map {i} on processor {name}')
        
        if args.map_disk: 
            m_vector = np.concatenate(mask*hp.read_map(f'{args.workdir}/healpy_maps/E/map_{i}.fits', field=[1,2], verbose=False, dtype=np.float64))
        else:
            cl = make_cl(args, comm, 'E')
            m_vector = np.concatenate(mask*hp.synfast(cl, args.nside, lmax=3*args.nside-1, pol=True, new=True)[1:])
        X_E = sparse.vstack((X_E, m_vector))
        
        if args.map_disk:
            m_vector = np.concatenate(mask*hp.read_map(f'{args.workdir}/healpy_maps/B/map_{i}.fits', field=[1,2], verbose=False, dtype=np.float64))
        else:
            cl = make_cl(args, comm, 'B')
            m_vector = np.concatenate(mask*hp.synfast(cl, args.nside, lmax=3*args.nside-1, pol=True, new=True)[1:])
        X_B = sparse.vstack((X_B, m_vector))
    
    X_E = X_E.tocsr()[1:]
    X_B = X_B.tocsr()[1:]
    
    C_E = sparse_covariance(X_E)
    C_B = sparse_covariance(X_B) 
    
    if rank == 0:
        print('On each RANK:')
        print(f'Data is {X_E.nnz/X_E.shape[1]/X_E.shape[0]*100}% filled')
        print(f'Size is {(X_E.data.nbytes + X_E.indptr.nbytes + X_E.indices.nbytes)/1e6} MB')
        print(f'Covariance is {C_E.nnz/C_E.shape[0]**2*100}% filled')
        print(f'Size is {(C_E.data.nbytes + C_E.indptr.nbytes + C_E.indices.nbytes)/1e6} MB')

        
    return C_E, C_B
    
#@timer
def main():    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    parser = argparse.ArgumentParser()
    add_cov_args(parser)
    args = parser.parse_args()
    
    hp.disable_warnings()
    C_E, C_B = make_cov(args, comm)

    C_E = comm.reduce(C_E, op=MPI.SUM, root=0)
    C_B = comm.reduce(C_B, op=MPI.SUM, root=0)
    
    if rank == 0:
        C_E = C_E/size
        C_B = C_B/size
            
        sparse.save_npz(f'{args.workdir}/C_E.npz', C_E)
        sparse.save_npz(f'{args.workdir}/C_B.npz', C_B)
        
if __name__ == "__main__":
    main()
    
    
    