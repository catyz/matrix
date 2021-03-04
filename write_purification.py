import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n',
        type=np.int,
    )

    args = parser.parse_args()

    with open('B_eig_1000.pkl', 'rb') as f:
        B_eigs, B_v = pickle.load(f)

    # with open('E_eig_500.pkl', 'rb') as handle:
    #     E_eigs, E_v = pickle.load(handle)
    
    B_eigs = B_eigs[-args.n:]
    B_v = B_v[:,-args.n:]
    print(f'Using the largest {args.n} eigenvectors')
    print(f'Largest eigenvalue: {B_eigs[-1]}')
    print(f'Smallest eigenvalue: {B_eigs[0]}')
    
    print('Setting small values to 0')
    B_v[np.where(abs(B_v)<1e-10)[0]] = 0   
    
    print('Constructing projection')
    B_v = sparse.csc_matrix(B_v)
    pure_B = B_v.dot(linalg.inv(B_v.T.dot(B_v)).dot(B_v.T))
    
    sparse.save_npz(f'pure_B_{args.n}.npz', pure_B)
    
    print('All done')

if __name__ == "__main__":
    main()
    
    
    
# def sparse_outer(v):
#     N = v.shape[0]
#     indices = np.where( abs(v)>1e-8 )[0]

#     data = []
#     row = []
#     col = []

#     for i in indices:
#         for j in indices:
#             row.append(i)
#             col.append(j)
#             data.append(v[i] * v[j])

#     pure = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N), dtype=np.float64)

#     return pure.tocsr() 

# def main():
#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         '--n',
#         type=np.int,
#     )

#     args = parser.parse_args()
    
#     with open('B_eig_500.pkl', 'rb') as handle:
#         B_eigs, B_v = pickle.load(handle)

#     # with open('E_eig_500.pkl', 'rb') as handle:
#     #     E_eigs, E_v = pickle.load(handle)

#     #n = len(np.where(B_eigs>1e3)[0])
#     print(args.n)

#     N = B_v.shape[0]
#     #pure_E = scipy.sparse.csr_matrix((N, N), dtype=np.float64)
#     pure_B = scipy.sparse.csr_matrix((N, N), dtype=np.float64)

#     for i in range(args.n):
#         if i % 150 == 0:
#             print(f'Processing {i}')

#         v = B_v[:, -i-1]
#         norm = np.linalg.norm(v)
#         v_norm = v/norm

#     #    assert eigs[i] < 1-1e-8 and eigs[-i-1] > 1+1e-8
#     #    pure_E += sparse_outer(v[:,i])
#         pure_B += sparse_outer(v_norm)

#     # with open('pure_E.pkl', 'wb') as handle:
#     #     pickle.dump(pure_E, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #     with open(f'pure_B_{args.n}.pkl', 'wb') as handle:
# #         pickle.dump(pure_B, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #    scipy.sparse.save_npz('pure_E.npz', pure_E)
#     scipy.sparse.save_npz('pure_B.npz', pure_B)

#     print("all done")