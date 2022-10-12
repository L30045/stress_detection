#%% import library
import os
import pickle
import scipy as sp
from scipy.io import loadmat
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas, plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as cm_display
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns

#%% load data
filename = 'ch_power.mat'
filepath = os.path.join('../data/classroom', filename)
psd_data = loadmat(filepath)
ch_psd_lib = psd_data['ch_psd_lib'][0]
chan30 = [x[0] for x in psd_data['chan30'][0]]
freqs = psd_data['freqs'][0]
label_increase = psd_data['label_increase']
label_normal = psd_data['label_normal']
sess_list = psd_data['sess_list'].T
y_true = np.array([True]*len(label_increase)+[False]*len(label_normal))

# LDA
# X_train, X_test, y_train, y_test = train_test_split(ch_psd_lib, y_true, test_size = 0.3)
# select feature frequencies
f_feat = np.arange(3,8)
f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
# select feature channels
ch_feat = ['FZ','FCZ','CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])

def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T
# define LDA classifier
solver = 'eigen'
shrinkage = None
is_weighted = True

# LOO
loo = LeaveOneOut()
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
acc = [False]*feat_lib.shape[0]
pred = np.zeros((len(acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib)):
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    # weighted
    if is_weighted:        
        priors = np.array([sum(y_train)/len(y_train), sum(~y_train)/len(y_train)])
        # priors = np.array([sum(y_true)/len(y_true),sum(~y_true)/len(y_true)])
    else:
        priors = None
    clf = LDA(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(X_train,y_train)
    pred[i] = clf.predict(X_test)
    acc[i] = clf.score(X_test,y_test)
print(f"LOO Acc. = {np.mean(acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(y_true,pred)*100:.2f}%")
cm_display.from_predictions(y_true,pred,normalize='true')


# %%
