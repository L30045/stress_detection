#%% import library
import os
import pickle
import scipy as sp
from scipy.io import loadmat
import numpy as np
import sklearn as sk
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut, LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas, plot_confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay as cm_display
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pandas as pd

#%% load data
filename = 'ch_power_proj_11Ch.mat'
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
# f_feat = np.arange(4,11)
f_feat = np.arange(3,8)
f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
# select feature channels
ch_feat = ['FZ', 'FCZ', 'CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])

def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T

# sig_feat = []

# for x in ch_psd_lib:
#     fz_feat = x[4,4:11]
#     fcz_feat = x[9,5:11]
#     cz_feat = x[14,6:8]
#     sig_feat.append(np.hstack([fz_feat,fcz_feat, cz_feat]))

# feat_lib = np.vstack(sig_feat)

# LOO
loo = LeaveOneOut()
# loo = LeaveOneGroupOut()
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
acc = [False]*feat_lib.shape[0]
pred = np.zeros((len(acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib,groups=sess_list.flatten())):
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
    # acc[i] = accuracy_score(pred[i],y_test)
    
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

#%% Leave-one-subject out
filename = 'ch_power_raw.mat'
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
subj_data = pd.read_csv(os.path.join('../data/classroom', 'firstsem_summary.csv'))
subj_list = np.array(subj_data['Subject Number'])
subj_list = np.vstack([subj_list[label_increase-1],subj_list[label_normal-1]])
nb_trial = [np.sum(subj_list==x) for x in np.unique(subj_list)]
f_feat = np.arange(3,8)
f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
ch_feat = ['FZ', 'FCZ', 'CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])
def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T
# LOO
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
acc = np.zeros((len(np.unique(subj_list)),1))
bacc = np.zeros(acc.shape)
f1 = np.zeros(acc.shape)
pred_lib = []
true_lib = []
for i,test_subj in enumerate(np.unique(subj_list)):
    test_idx = np.squeeze(subj_list==test_subj)
    if test_idx==[]:
        continue
    train_idx = ~test_idx
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    # SMOTE
    if is_smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # weighted
    if is_weighted:        
        w_inc = len(y_train)/sum(y_train)
        n_inc = len(y_train)/sum(~y_train)
        X_train[y_train,:] *= w_inc
        X_train[~y_train,:] *= n_inc
    # prior
    if is_prior:        
        priors = [0.5,0.5]
    else:
        priors = None
    clf = LDA(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    pred_lib.append(pred)
    true_lib.append(y_test)
    acc[i] = clf.score(X_test,y_test)
    bacc[i] = bas(y_test,pred)
    f1[i] = f1_score(y_test,pred)

print(f"Mean LOO Acc. = {np.mean(acc)*100:.2f}%")
print(f"Mean LOO Balanced Acc. = {np.mean(bacc)*100:.2f}%")
print(f"Median LOO Balanced Acc. = {np.median(bacc)*100:.2f}%")
print(f"Mean LOO F1 score = {np.mean(f1):.2f}")

plt.figure()
# plt_bacc = sorted(zip(bacc,nb_trial),key=lambda x: x[1],reverse=True)
# plt.bar(np.unique(subj_list),[x[0][0] for x in plt_bacc])
plt.bar(np.unique(subj_list),bacc.reshape(-1))
plt.xlabel('Subject ID')
plt.ylabel('Balanced Acc.')
plt.grid()

cm_display.from_predictions(true_lib[0],pred_lib[0],normalize=None,display_labels=['Normal','Increase'])

#%%
# find all increase subject
all_inc = []
all_norm = []
mix_subj = []
plt_inc = []
plt_norm = []
inc_subj = subj_list[:len(label_increase)]
norm_subj = subj_list[len(label_increase):]
color_list = []
for subj_i in np.unique(subj_list):
    plt_inc.append(sum(inc_subj==subj_i))
    plt_norm.append(sum(norm_subj==subj_i))
    if sum(inc_subj==subj_i)==0:
        all_norm.append(subj_i)
        flag = 0
    elif sum(norm_subj==subj_i)==0:
        all_inc.append(subj_i)
        flag = 1
    else:
        mix_subj.append(subj_i)
    if flag==1:
        color_list.append('r')
    elif flag==0:
        color_list.append('b')
    else:
        color_list.append('g')
    flag=-1

# number of session per subject
plt.figure()
plt.bar(np.unique(subj_list),nb_trial,color=color_list)
plt.xlabel('Subject ID')
plt.ylabel('# of Sessions')
plt.xticks(np.unique(subj_list))
plt.grid()

plt.figure()
plt.bar(np.unique(subj_list),bacc.reshape(-1),color=color_list)
plt.xlabel('Subject ID')
plt.ylabel('Balanced Acc.')
plt.xticks(np.unique(subj_list))
plt.grid()

plt.figure(figsize=(10,8))
plt.bar(np.unique(subj_list),np.array(plt_inc).reshape(-1),color='r',bottom=np.array(plt_norm).reshape(-1),label='Increased')
plt.bar(np.unique(subj_list),np.array(plt_norm).reshape(-1),color='b',label='Normal')
plt.xticks(np.unique(subj_list))
plt.grid()
plt.xlabel('Subject ID',fontsize=20)
plt.ylabel('# of Sessions',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.legend(fontsize=12)

#%%
# plt.figure()
# x_ruler = np.vstack([np.unique(subj_list)-0.2,np.unique(subj_list)+0.2]).T.reshape(-1)
# bardata = np.hstack([bacc_raw,bacc_11]).reshape(-1)
# color_list = ['b','r']*18
# plt.bar(x_ruler, bardata, width=0.3, color=color_list)

sub_bacc_raw = bacc_raw[[2,3,8,10,14,15]]*100
sub_bacc_11 = bacc_11[[2,3,8,10,14,15]]*100
plt.figure(figsize=(8,6))
barwidth = 0.4
x_ruler = np.vstack([np.arange(1-barwidth/2,7-barwidth/2),np.arange(1+barwidth/2,7+barwidth/2)]).T.reshape(-1)
bardata = np.hstack([sub_bacc_raw,sub_bacc_11]).reshape(-1)
color_list = [[1,1,1],[0,0,1]]*18
plt.bar(np.arange(1-barwidth/2,7-barwidth/2), sub_bacc_raw.reshape(-1), width=barwidth, color=[0,0.5,0])
plt.bar(np.arange(1+barwidth/2,7+barwidth/2), sub_bacc_11.reshape(-1), width=barwidth, color=[0,0.8,0])
plt.xlabel('Subject ID',fontsize=20)
plt.ylabel('Balanced Acc.',fontsize=20)
plt.xticks(np.arange(1,7),labels=mix_subj,fontsize=15)
plt.yticks(fontsize=15)
plt.grid()
plt.legend(['BP only','BP+ASR+Projection'],fontsize=12)


#%% Leave-one-subject out on only mixed subject
filename = 'ch_power_proj_11Ch.mat'
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
subj_data = pd.read_csv(os.path.join('../data/classroom', 'firstsem_summary.csv'))
subj_list = np.array(subj_data['Subject Number'])
subj_list = np.vstack([subj_list[label_increase-1],subj_list[label_normal-1]])
y_true = y_true[[x in [3,4,9,11,16,17] for x in subj_list]]
nb_trial = [np.sum(subj_list==x) for x in np.unique(subj_list)]
f_feat = np.arange(3,8)
f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
ch_feat = ['FZ', 'FCZ', 'CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])
def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T
# LOO
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
feat_lib = feat_lib[[x in [3,4,9,11,16,17] for x in subj_list],:]
subj_list = subj_list[[x in [3,4,9,11,16,17] for x in subj_list]]
acc = np.zeros((len(np.unique(subj_list)),1))
bacc = np.zeros(acc.shape)
f1 = np.zeros(acc.shape)
for i,test_subj in enumerate(np.unique(subj_list)):
    test_idx = np.squeeze(subj_list==test_subj)
    if test_idx==[]:
        continue
    train_idx = ~test_idx
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    # SMOTE
    if is_smote:
        X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    # weighted
    if is_weighted:        
        w_inc = len(y_train)/sum(y_train)
        n_inc = len(y_train)/sum(~y_train)
        X_train[y_train,:] *= w_inc
        X_train[~y_train,:] *= n_inc
    # prior
    if is_prior:        
        priors = [0.5,0.5]
    else:
        priors = None
    clf = LDA(solver=solver, shrinkage=shrinkage, priors=priors)
    clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    acc[i] = clf.score(X_test,y_test)
    bacc[i] = bas(y_test,pred)
    f1[i] = f1_score(y_test,pred)

print(f"Mean LOO Acc. = {np.mean(acc)*100:.2f}%")
print(f"Mean LOO Balanced Acc. = {np.mean(bacc)*100:.2f}%")
print(f"Median LOO Balanced Acc. = {np.median(bacc)*100:.2f}%")
print(f"Mean LOO F1 score = {np.mean(f1):.2f}")
plt.figure()
plt.bar([1,2,3,4,5,6],bacc.reshape(-1))
plt.xlabel('Subject ID')
plt.ylabel('Balanced Acc.')
plt.xticks([1,2,3,4,5,6],[3,4,9,11,16,17])
plt.grid()