import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import pymaster as nmt
import scipy.sparse as sparse

hp.disable_warnings()

mask = hp.read_map('south_patch_apo_64.fits',verbose=False, dtype=np.float64)
pure_B = sparse.load_npz('pi_B_2000.npz')
#ure_B = np.load('pi_B_numpy.npy')

nreal = 100
nside = hp.get_nside(mask)
lmax = 3*nside-1
npix = hp.nside2npix(nside)
b = nmt.NmtBin.from_nside_linear(nside, 32, is_Dell=True)
ell = b.get_effective_ells()
np.save('ell.npy', ell)

ell_th, TT, EE, BB, TE = np.loadtxt('totcls_lensed_r0p1.txt', unpack=True)

f_pure = nmt.NmtField(mask=mask, maps=hp.read_map('healpy_maps/map_0.fits', field=[1,2], verbose=False, dtype=np.float64), n_iter_mask_purify = 10, purify_b = True) #beam = b_ell,
f_mat = nmt.NmtField(mask=mask, maps=hp.read_map('healpy_maps/map_0.fits', field=[1,2], verbose=False, dtype=np.float64) ) #beam = b_ell

w_pure = nmt.NmtWorkspace()
w_pure.compute_coupling_matrix(f_pure, f_pure, b)

w_mat = nmt.NmtWorkspace()
w_mat.compute_coupling_matrix(f_mat, f_mat, b)

cl_th = np.array([EE, np.zeros_like(EE), np.zeros_like(EE), BB])
cl_th_binned = w_pure.decouple_cell(w_pure.couple_cell(cl_th))

prefactor = 2*np.pi/ell/(ell+1)
cl_th_binned *= prefactor

cl_mean = np.zeros((4, len(ell)))
cl_std = np.zeros((4, len(ell)))

pure_cl_mean = np.zeros((4, len(ell)))
pure_cl_std = np.zeros((4, len(ell)))

for i in range(nreal):
    
    #KS PURIFICATION
    f_pure = nmt.NmtField(mask=mask, maps=hp.read_map(f'healpy_maps/map_{i}.fits', field=[1,2], verbose=False, dtype=np.float64), n_iter_mask_purify=10, purify_b=True) #, beam = b_ell
    cl = w_pure.decouple_cell(nmt.compute_coupled_cell(f_pure, f_pure))
    cl_mean += cl
    cl_std += cl*cl
    
    #MAT PURIFICATION
    m = mask*hp.read_map(f'healpy_maps/map_{i}.fits', field=[1,2], verbose=False, dtype=np.float64)
    m_pure = pure_B.dot(np.concatenate(m)).reshape(2, npix)
    f_mat = nmt.NmtField(mask=mask, maps=[m_pure[0], m_pure[1]]) #beam = b_ell
    
    cl = w_mat.decouple_cell(nmt.compute_coupled_cell(f_mat, f_mat))
    pure_cl_mean += cl
    pure_cl_std += cl*cl

    
cl_mean /= nreal
cl_std = np.sqrt(cl_std / nreal - cl_mean*cl_mean)

pure_cl_mean /= nreal
pure_cl_std = np.sqrt(pure_cl_std / nreal - pure_cl_mean*pure_cl_mean)

np.save('KS_cl_mean.npy', cl_mean)
np.save('KS_cl_std.npy', cl_std)
np.save('matrix_cl_mean.npy', pure_cl_mean)
np.save('matrix_cl_std.npy', pure_cl_std)