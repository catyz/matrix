import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import pickle
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--n',
    type=np.int,
    required=True,
    )

args = parser.parse_args()

print(args.n)

# with open('C_E.pkl', 'rb') as handle:
#     C_E = pickle.load(handle)
    
# with open('C_B.pkl', 'rb') as handle:
#     C_B = pickle.load(handle)

C_E = sparse.load_npz('C_E.npz')
C_B = sparse.load_npz('C_B.npz')

sigma = np.mean(C_E.diagonal())/1000
print(sigma)
factor = sparse.identity(C_E.shape[0], format='csr') * sigma

B_eig = linalg.eigsh(C_B+factor, args.n, C_E+factor, which='LM', return_eigenvectors=True)
#E_eig = linalg.eigsh(C_E+factor, args.n, C_B+factor, which='LM', return_eigenvectors=True)

with open(f'B_eig_{args.n}.pkl', 'wb') as handle:
    pickle.dump(B_eig, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
# with open(f'E_eig_{args.n}.pkl', 'wb') as handle:
#     pickle.dump(E_eig, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("all done")