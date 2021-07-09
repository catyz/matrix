import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pickle
import argparse
from mpi4py import MPI
from glob import glob
import os
from write_data import timer, add_data_args
    
@timer
def center_data(X):
    print('Centering data...')
    X = X.transpose()
    n = X.shape[1]
    means = X.mean(axis=1)
    nz = np.where(means!=0)[0]
    
    for i in nz:
        centering = np.ones((1,n))*means[i,0]
        X[i] -= centering
        
    return X
    
@timer
def centered_cov(X):
    print('Covariance...')
    n = X.shape[1]
    C = X.dot(X.transpose())/(n-1)
    
    return C.tocsr()

@timer
def make_cov(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    name = MPI.Get_processor_name()
    
    npix = 12*args.nside**2
    m_vector = np.zeros(2*npix)

    if rank == 0:
        X_E = sparse.load_npz(f'{args.workdir}/X_E.npz')
        print(f'Data shape is {X_E.shape}')
        X_E = center_data(X_E)
        C_E = centered_cov(X_E)
        print(f'Covariance is {C_E.shape[1]} by {C_E.shape[0]}, {C_E.nnz/C_E.shape[1]/C_E.shape[0]*100}% filled')
        print(f'Size is {(C_E.data.nbytes + C_E.indptr.nbytes + C_E.indices.nbytes)/1e9} GB')
        
        sparse.save_npz(f'{args.workdir}/C_E.npz', C_E)

    if rank ==1:
        X_B = sparse.load_npz(f'{args.workdir}/X_B.npz')
        print(f'Data shape is {X_B.shape}')
        X_B = center_data(X_B)
        C_B = centered_cov(X_B)

        sparse.save_npz(f'{args.workdir}/C_B.npz', C_B)
        print('INFO: FINISHED COVARIANCE')
        
@timer
def main():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    args = parser.parse_args()
    
    make_cov(args)

if __name__ == "__main__":
    main()