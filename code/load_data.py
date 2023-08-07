#%%
import os
from scipy.io import loadmat
import numpy as np

#%%
def load_data(filename):
    """load data helper

    Args:
        filename (string): filename

    Returns:
        feat_lib: (samples, features matrix)
        y_true: true labels (samples,)
    """
    filepath = os.path.join('../data', filename)
    psd_data = loadmat(filepath)
    ch_psd_lib = psd_data['ch_psd_lib'][0]
    chan30 = [x[0] for x in psd_data['chan30'][0]]
    freqs = psd_data['freqs'][0]
    label_increase = psd_data['label_increase']
    label_normal = psd_data['label_normal']
    sess_list = psd_data['sess_list'].T
    y_true = np.array([True]*len(label_increase)+[False]*len(label_normal))
    f_feat = np.arange(3,8)
    f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
    # select feature channels
    ch_feat = ['FZ', 'FCZ', 'CZ']
    ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])

    def select_feat(ch_psd_lib, f_idx, ch_idx):
        feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
        return feat_lib.T

    feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
    
    return feat_lib, y_true
    
    
    
    