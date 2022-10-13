#%%
import numpy as np
import scipy.io as sio

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, plot_confusion_matrix, f1_score, matthews_corrcoef, mean_squared_error, recall_score, precision_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet, HuberRegressor
from sklearn.utils import class_weight


from xgboost import XGBClassifier

from matplotlib import pyplot as plt
import seaborn as sns

import warnings

#%%
nfold = 5
balanced_acc = np.zeros((nfold,))

f_feature = np.arange(4, 9)

for f_i in range(nfold):
#for f_i in range(1):
    fold_mat = sio.loadmat(f'../icawm_allCh_fold{f_i+1}.mat')
    norm_X = fold_mat['plt_medall_norm']
    norm_X = norm_X[:, f_feature, :]
    norm_X = np.reshape(norm_X, (norm_X.shape[0], -1))
    inc_X = fold_mat['plt_medall_inc']
    inc_X = inc_X[:, f_feature, :]
    inc_X = np.reshape(inc_X, (inc_X.shape[0], -1))

    test_norm_idx = fold_mat['test_norm_idx']
    test_inc_idx = fold_mat['test_inc_idx']

    label_increase = fold_mat['label_increase']
    label_normal = fold_mat['label_normal']

    testnorm = np.intersect1d(label_normal, test_norm_idx, return_indices=True)[1]
    testinc = np.intersect1d(label_increase, test_inc_idx, return_indices=True)[1]

    test_X = np.concatenate((norm_X[testnorm, :], inc_X[testinc, :]), axis=0)
    test_y = np.concatenate((np.zeros(testnorm.shape[0],), np.ones(testinc.shape[0],)))
    train_X = np.concatenate((np.delete(norm_X, testnorm, axis=0), np.delete(inc_X, testinc, axis=0)), axis=0)
    train_y = np.concatenate((np.zeros((norm_X.shape[0] - testnorm.shape[0],)), np.ones((inc_X.shape[0] - testinc.shape[0]))))

    lda_clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto', priors=[0.5, 0.5])
    lda_clf.fit(train_X, train_y)
    lda_py = lda_clf.predict(test_X)

    balanced_acc[f_i] = balanced_accuracy_score(test_y, lda_py)

print(f'Bacc: {np.mean(balanced_acc) * 100:.2f}%')


    
# %%
