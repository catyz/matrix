import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pickle
import argparse
from glob import glob
import os
from write_data import timer, add_data_args

def add_eigs_args(parser):
    parser.add_argument(
        '-e',
        type=int,
        required=False,
        default=1000,
        help='Number of eigenvectors to compute'
    )

@timer
def solve_eigs(args, C_E, C_B):
    print(f'Solving {args.e} eigs...')
    sigma = np.mean(C_E.diagonal())/100
    print(f'Sigma: {sigma}')
    factor = sparse.identity(C_E.shape[0], format='csr') * sigma
    B_eigs = linalg.eigsh(C_B+factor, args.e, C_E+factor, which='LM', return_eigenvectors=True)
    return B_eigs

def main():
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_eigs_args(parser)
    args = parser.parse_args()
    
    print('Loading covariances...')
    C_E = sparse.load_npz(f'{args.workdir}/C_E.npz')
    C_B = sparse.load_npz(f'{args.workdir}/C_B.npz')

    B_eigs = solve_eigs(args, C_E, C_B)
    print(f'Eig range is from {B_eigs[0][0]} to {B_eigs[0][-1]}')
    
    if B_eigs[0][0] > 1.02:
        print('WARNING: smallest eigenvalue is greater than cut threshold. Re-run with higher k')
    
    with open(f'{args.workdir}/B_eigs_{args.e}.pkl', 'wb') as f:
        pickle.dump(B_eigs, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('INFO: FINISHED EIGS')
        
if __name__ == "__main__":
    main()