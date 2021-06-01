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

def add_data_args(parser):
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
        default=16,
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
        '--mask',
        required=False,
        default='/scratch/yuyang/matrix/south_patch_apo_64.fits',
        help='Path to mask'
    )
    
    parser.add_argument(
        '--workdir',
        required=False,
        default='/scratch/yuyang/matrix',
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
    
def remove_nans(m_vector):
    nan_indx = np.where(np.isnan(m_vector))[0]
    if len(nan_indx) !=0:
        #print(f'nans at {nan_indx}')
        m_vector[nan_indx] = 0
        
#@timer
def write_distributed_data(args, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    
    nreal = args.nreal
    nside = args.nside
    npix = hp.nside2npix(nside)
    
    if rank == 0:
        #mask = hp.ud_grade(hp.read_map(args.mask, verbose=False, dtype=np.float64), nside_out=args.nside)
        #mask = hp.read_map(args.mask, verbose=False, dtype=np.float64)     
        mask = np.ones(npix)
        reals = np.arange(nreal)
        chunks = np.array_split(reals, size)
        m_vector = np.zeros(2*npix)
        X_E = sparse.coo_matrix(m_vector, dtype=np.float64)
        X_B = sparse.coo_matrix(m_vector, dtype=np.float64)
        del m_vector
    else:
        chunks = None
        mask = None
        X_E = None
        X_B = None
    
    X_E = comm.bcast(X_E, root=0)
    X_B = comm.bcast(X_B, root=0)
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
        
    sparse.save_npz(f'{args.workdir}/X/X_E_{chunk[0]}-{chunk[-1]}.npz', X_E)
    sparse.save_npz(f'{args.workdir}/X/X_B_{chunk[0]}-{chunk[-1]}.npz', X_B)

    if rank == 0:
        print('On each RANK:')
        print(f'Data is {X_E.nnz/X_E.shape[1]/X_E.shape[0]*100}% filled')
        print(f'Size is {(X_E.data.nbytes + X_E.indptr.nbytes + X_E.indices.nbytes)/1e6} MB')
    
    comm.Barrier()

def gather_data(args, comm):
    rank = comm.Get_rank()
    npix = 12* args.nside**2
    
    if rank == 0:
        m_vector = np.zeros(2*npix)
        X_E = sparse.coo_matrix(m_vector)
        X_B = sparse.coo_matrix(m_vector)
        del m_vector
        
        for file in glob(f'{args.workdir}/X/*'):
            if 'X_E' in file:
                print(f'Gathering {file}')
                X_chunk = sparse.load_npz(file)
                X_E = sparse.vstack((X_E, X_chunk))            
            if 'X_B' in file:
                print(f'Gathering {file}')
                X_chunk = sparse.load_npz(file)
                X_B = sparse.vstack((X_B, X_chunk))
            os.remove(file)

        X_E = X_E.tocsr()[1:]
        X_B = X_B.tocsr()[1:]
        
        print('Finished gathering, writing data matrix to disk...')
        sparse.save_npz(f'{args.workdir}/X_E.npz', X_E)
        sparse.save_npz(f'{args.workdir}/X_B.npz', X_B)
        print('INFO: FINISHED DATA')


#@timer
def main():    
    comm = MPI.COMM_WORLD
    
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    args = parser.parse_args()
    print(f'INFO: NSIDE {args.nside}')
    
    hp.disable_warnings()
    write_distributed_data(args, comm)
    gather_data(args, comm)
        
if __name__ == "__main__":
    main()
    
    
    