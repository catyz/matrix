import numpy as np
import healpy as hp
import scipy.sparse as sparse
import pymaster as nmt

def sparse_covariance(X):
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
        
    n = X.shape[0]
    means = np.array(X.mean(axis=0))[0]
    nz = np.where(means!=0)[0]

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
    

def make_data():
    nreal = 1700
    nside = 128
    npix = hp.nside2npix(nside)

    mask = hp.read_map('./toast_maps/0/lcdm_telescope_all_time_all_invnpp.fits',verbose=False, dtype=np.float64)
    mask[np.where(mask!=0)] = 1
    mask = nmt.mask_apodization(mask, 5, apotype='C1')
    print('Apodization degree 5')
    
    m_vector = np.concatenate(mask*hp.read_map('./pure_maps/E/E_map_0.fits', field=[1,2], verbose=False, dtype=np.float64))
#     m_vector = np.concatenate(mask_apo*hp.read_map('./out_pure_maps/0/E_telescope_all_time_all_binned.fits', field=[1,2], verbose=False, dtype=np.float64))
#    remove_nans(m_vector)
    X_E = sparse.coo_matrix(m_vector)
    
    m_vector = np.concatenate(mask*hp.read_map('./pure_maps/B/B_map_0.fits', field=[1,2], verbose=False, dtype=np.float64))
#     m_vector = np.concatenate(mask_apo*hp.read_map('./out_pure_maps/0/B_telescope_all_time_all_binned.fits', field=[1,2], verbose=False, dtype=np.float64))
#    remove_nans(m_vector)
    X_B = sparse.coo_matrix(m_vector)

    for i in range(nreal-1):
        if i % 100 == 0:
            print(f'Processing map {i}')
            
        m_vector = np.concatenate(mask*hp.read_map(f'./pure_maps/E/E_map_{i+1}.fits', field=[1,2], verbose=False, dtype=np.float64))
#         m_vector = np.concatenate(mask_apo*hp.read_map(f'./out_pure_maps/{i+1}/E_telescope_all_time_all_binned.fits', field=[1,2], verbose=False, dtype=np.float64))
#        remove_nans(m_vector)
        X_E = sparse.vstack([X_E, m_vector])
        
        m_vector = np.concatenate(mask*hp.read_map(f'./pure_maps/B/B_map_{i+1}.fits', field=[1,2], verbose=False, dtype=np.float64))
#         m_vector = np.concatenate(mask_apo*hp.read_map(f'./out_pure_maps/{i+1}/B_telescope_all_time_all_binned.fits', field=[1,2], verbose=False, dtype=np.float64))
#        remove_nans(m_vector)
        X_B = sparse.vstack([X_B, m_vector])
    
    return X_E.tocsr(), X_B.tocsr()
    
def main():
    X_E, X_B = make_data()
    print('Finished making data')
    
    C_E = sparse_covariance(X_E)
    C_B = sparse_covariance(X_B)
    print('Finished covariance')
    
    sparse.save_npz('C_E.npz', C_E)
    sparse.save_npz('C_B.npz', C_B)
    print('Saved covariance')
    
if __name__ == "__main__":
    main()
    
    
    
# def covariance(m_q, m_u):
#     assert len(m_q) == len(m_u)
#     assert np.array_equal(np.where(m_q!=0)[0], np.where(m_u!=0)[0]) == True
    
#     N = len(m_q)
#     indices = np.where(m_q!=0)[0]
#     mean_q = np.mean(m_q)
#     mean_u = np.mean(m_u)

#     data_qq = []
#     data_qu = []
#     data_uu = []
    
#     row = []
#     col = []

#     for i in indices:
#         for j in indices:
#             row.append(i)
#             col.append(j)
#             data_qq.append((m_q[i]-mean_q) * (m_q[j]-mean_q))
#             data_qu.append((m_q[i]-mean_q) * (m_u[j]-mean_u))
#             data_uu.append((m_u[i]-mean_u) * (m_u[j]-mean_u))

#     c_qq = scipy.sparse.coo_matrix((data_qq, (row, col)), shape=(N, N), dtype=np.float64)
#     c_qu = scipy.sparse.coo_matrix((data_qu, (row, col)), shape=(N, N), dtype=np.float64)
#     c_uq = c_qu.transpose()
#     c_uu = scipy.sparse.coo_matrix((data_uu, (row, col)), shape=(N, N), dtype=np.float64)
            
#     top = scipy.sparse.hstack([c_qq, c_qu])
#     bottom = scipy.sparse.hstack([c_uq, c_uu])
#     C = scipy.sparse.vstack([top, bottom])

#     return C.tocsr()

# def main():
#     nreal = 1653
#     nside = 128
#     npix = hp.nside2npix(nside)
#     mask = hp.read_map('./out_maps/0/lcdm_telescope_all_time_all_invnpp.fits',verbose=False, dtype=np.float64)
#     mask[np.where(mask!=0)] = 1

#     C_E = scipy.sparse.csr_matrix((2*npix, 2*npix), dtype=np.float64)
#     C_B = scipy.sparse.csr_matrix((2*npix, 2*npix), dtype=np.float64)

#     for i in range(nreal):
#         if i % 150 == 0:
#             print(f'Processing {i}')

#         m_q, m_u = mask*hp.read_map(f'./pure_maps/E/E_map_{i}.fits',field =[1,2], verbose=False, dtype=np.float64)
#         C_E += covariance(m_q, m_u)

#         m_q, m_u = mask*hp.read_map(f'./pure_maps/B/B_map_{i}.fits',field =[1,2], verbose=False, dtype=np.float64)
#         C_B += covariance(m_q, m_u)

#     C_E /= nreal
#     C_B /= nreal

#     scipy.sparse.save_npz('C_E.npz', C_E)
#     scipy.sparse.save_npz('C_B.npz', C_B)
    
# #     with open('C_E.pkl', 'wb') as handle:
# #         pickle.dump(C_E, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #     with open('C_B.pkl', 'wb') as handle:
# #         pickle.dump(C_B, handle, protocol=pickle.HIGHEST_PROTOCOL)

#     print("all done")