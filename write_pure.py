import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pickle
import argparse
import healpy as hp
from write_cov import timer, add_cov_args
from write_eigs import add_eigs_args

def add_pure_args(parser):
    parser.add_argument(
        '-n',
        required=False,
        type=int,
        default=500,
        help='Number of vectors to use'
    )

@timer
def load_eigs(args):
    print(f'Loading {args.e} eigenvectors')
    with open(f'{args.workdir}/B_eigs_{args.e}.pkl', 'rb') as f:
        B_eigs = pickle.load(f)
    return B_eigs

@timer
def small_to_zero(v):
    hp.disable_warnings()
    mask = hp.read_map('./toast_pure_maps/0/B_telescope_all_time_all_invnpp.fits',verbose=False, dtype=np.float64)
    mask = np.concatenate((mask, mask))
    zeros = np.where(mask==0)[0]
    print(f'{len(zeros)} zeros in each column')
    for i in range(v.shape[1]):
        v[:,i][zeros] = 0
    return v

@timer
def construct_purification(args, B_eigs):
    B_v = B_eigs[1]
    B_eigs = B_eigs[0]
    
    B_eig = B_eigs[-args.n:]
    B_v = B_v[:,-args.n:]
    print('Constructing purification...')
    print(f'Using the largest {args.n} eigenvectors')
    print(f'Largest eigenvalue: {B_eigs[-1]}')
    print(f'Smallest eigenvalue: {B_eigs[0]}') 
    B_v = small_to_zero(B_v)
    
    B_v = sparse.csr_matrix(B_v)
    pi_B = B_v.dot(linalg.inv(B_v.T.dot(B_v)).dot(B_v.T))
    
    #Check a few eigenvalues to make sure it's indeed a projection
#     eigs = linalg.eigsh(pi_B, 3, which='LM', return_eigenvectors=False)
#     print(f'First 3 eigenvalues of projection matrix: {eigs}')
    
    sparse.save_npz(f'pi_B_{args.n}.npz', pi_B)

def main():
    parser = argparse.ArgumentParser()
    add_cov_args(parser)
    add_eigs_args(parser)
    add_pure_args(parser)
    args = parser.parse_args()   
    
    B_eigs = load_eigs(args)
    construct_purification(args, B_eigs)
    
if __name__ == '__main__':
    main()