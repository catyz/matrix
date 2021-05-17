from scipy import sparse
import matplotlib.pyplot as plt
from write_data import timer
import numpy as np

@timer
def save_fig(C, name):
    fig = plt.figure()
    plt.imshow(C)
    fig.savefig(f'{name}')

@timer    
def load_cov(mode):
    C = sparse.load_npz(f'{mode}.npz')
    return C

@timer
def splice(C_E, C_B):
    nz = []
    for i, row in enumerate(C_E):
        if np.sum(row) != 0:
             nz.append(i)
    C_E_dense = C_E[nz].tocsc()[:,nz].todense()
    C_B_dense = C_B[nz].tocsc()[:,nz].todense()

    return C_E_dense, C_B_dense
    
def main():
    C_E = load_cov('C_E')
    C_B = load_cov('C_B')
    C_E_dense, C_B_dense = splice(C_E, C_B)
    save_fig(C_E_dense, 'C_E.png')
    save_fig(C_B_dense, 'C_B.png')

if __name__ == '__main__':
    main()