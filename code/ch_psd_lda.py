#%% import library
import os
import pickle
import scipy as sp
from scipy.io import loadmat
import numpy as np
import sklearn as sk
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas, plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay as cm_display
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

#%% load data
filename = 'ch_power.mat'
# define LDA classifier
solver = 'eigen'
shrinkage = False
is_weighted = False
is_prior = True
is_smote = False
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
ch_feat = ['FZ', 'FCZ', 'CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])

def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T

# LOO
loo = LeaveOneOut()
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
acc = [False]*feat_lib.shape[0]
pred = np.zeros((len(acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib)):
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    # SMOTE
    if is_smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # weighted
    if is_weighted:        
        w_inc = len(y_train)/sum(y_train)
        n_inc = len(y_train)/sum(~y_train)
        # w_prior = sum(y_train)/len(y_train)
        # n_prior = sum(~y_train)/len(y_train)
        # w_inc = w_prior/w_inc
        # n_inc = n_prior/n_inc
        X_train[y_train,:] *= w_inc
        X_train[~y_train,:] *= n_inc
    # prior
    if is_prior:        
        # priors = np.array([sum(y_train)/len(y_train), sum(~y_train)/len(y_train)])
        # priors = np.array([sum(y_true)/len(y_true),sum(~y_true)/len(y_true)])
        priors = [0.5,0.5]
    else:
        priors = None
    clf = LDA(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(X_train,y_train)
    pred[i] = clf.predict(X_test)
    acc[i] = clf.score(X_test,y_test)
print(f"LOO Acc. = {np.mean(acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(y_true,pred)*100:.2f}%")
print(f"LOO F1 score = {f1_score(y_true,pred):.2f}")
cm_display.from_predictions(y_true,pred,normalize=None,display_labels=['Normal','Increase'])


# %% grid search parameters
X_train, y_train = feat_lib, y_true
grid = dict()
grid['shrinkage'] = np.arange(0, 1, 0.01)
# priors = np.array([sum(y_train)/len(y_train), sum(~y_train)/len(y_train)])
priors = [0.5,0.5]
search = GridSearchCV(LDA(solver=solver,priors=priors), grid, scoring='balanced_accuracy', cv=loo, n_jobs=-1)
results = search.fit(X_train,y_train)
# summarize
print('Mean Accuracy: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

# %%
max_acc = 0
max_bas = 0
max_f1 = 0

for shrinkage in [False, True]:
    for is_prior in [False, True]:
        for is_smote in [False, True]:
            # LOO
            loo = LeaveOneOut()
            feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
            acc = [False]*feat_lib.shape[0]
            pred = np.zeros((len(acc),1))
            for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib)):
                X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
                X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
                # SMOTE
                if is_smote:
                    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                # weighted
                if is_weighted:        
                    w_inc = len(y_train)/sum(y_train)
                    n_inc = len(y_train)/sum(~y_train)
                    # w_prior = sum(y_train)/len(y_train)
                    # n_prior = sum(~y_train)/len(y_train)
                    # w_inc = w_prior/w_inc
                    # n_inc = n_prior/n_inc
                    X_train[y_train,:] *= w_inc
                    X_train[~y_train,:] *= n_inc
                # prior
                if is_prior:        
                    # priors = np.array([sum(y_train)/len(y_train), sum(~y_train)/len(y_train)])
                    # priors = np.array([sum(y_true)/len(y_true),sum(~y_true)/len(y_true)])
                    priors = [0.5,0.5]
                else:
                    priors = None
                clf = LDA(solver=solver, shrinkage=shrinkage, priors=priors)
                clf.fit(X_train,y_train)
                pred[i] = clf.predict(X_test)
                acc[i] = clf.score(X_test,y_test)
            if bas(y_true,pred) > max_bas:
                max_acc = np.mean(acc)
                max_bas = bas(y_true,pred)
                max_f1 = f1_score(y_true,pred)
                max_set = [shrinkage, is_prior, is_smote]