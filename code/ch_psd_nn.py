#%%
import os
import pickle
import re
import scipy as sp
from scipy.io import loadmat
import numpy as np
import sklearn as sk
from sklearn.model_selection import GridSearchCV, train_test_split, LeaveOneOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import balanced_accuracy_score as bas, plot_confusion_matrix
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as cm_display
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
y_true = np.expand_dims(np.array([True]*len(label_increase)+[False]*len(label_normal)),1)
f_feat = np.arange(3,8)
f_idx = np.array([i for i,x in enumerate(freqs) if x in f_feat])
# select feature channels
ch_feat = ['FZ', 'FCZ', 'CZ']
ch_idx = np.array([i for i,x in enumerate(chan30) if x in ch_feat])
# ch_idx = np.arange(30)
def select_feat(ch_psd_lib, f_idx, ch_idx):
    feat_lib = np.hstack([x[np.ix_(ch_idx,f_idx)].reshape(-1,1) for x in ch_psd_lib])
    return feat_lib.T
feat_lib = select_feat(ch_psd_lib,f_idx,ch_idx)
feat_lib = torch.tensor(feat_lib,dtype=torch.float)
y_true = torch.tensor(y_true,dtype=torch.float)

#%% define model
# TO DO LIST: Reduce layers. Make a better slides
class simple_nn(nn.Module):
    def __init__(self) -> None:
        super(simple_nn, self).__init__()
        self.flatten = nn.Flatten()
        self.nn_struct = nn.Sequential(
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5,1),
            nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.nn_struct[0].weight.data,nonlinearity='relu')
        nn.init.kaiming_normal_(self.nn_struct[2].weight.data,nonlinearity='relu')
        nn.init.kaiming_normal_(self.nn_struct[4].weight.data,nonlinearity='relu')
    
    def forward(self,x):
        x = self.flatten(x)
        out = self.nn_struct(x)
        return out

#%% define loss function and optimizer
loss_fn = nn.BCELoss()
def train(X, y, model, loss_fn, optimizer):
    model.train()
    X, y = X.to(device), y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, y)
    train_acc = (pred.reshape(-1).detach().numpy().round() == y.reshape(-1).detach().numpy()).mean()

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return train_acc, loss.item()

#%% define test
def test(X, y, model, loss_fn):
    model.eval()
    with torch.no_grad():
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss = loss_fn(pred, y).item()
        correct = (pred.reshape(-1).detach().numpy().round() ==
                   y.reshape(-1).detach().numpy()).mean()
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
#%% Train model without SMOTE
# LOO
loo = LeaveOneOut()
epochs = 1000
train_acc = []
train_loss = []
train_bas = []
test_acc = [False]*feat_lib.shape[0]
pred = np.zeros((len(test_acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib)):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = simple_nn().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    for t in range(epochs):
        acc, loss = train(X_train, y_train, model, loss_fn, optimizer)
        if i == 0:
            train_acc.append(acc)
            train_loss.append(loss)
            model.eval()
            train_pred = model(X_train).detach().numpy().round()
            train_bas.append(bas(train_pred,y_train.reshape(-1).detach().numpy()))
    model.eval()
    pred[i] = model(X_test).detach().numpy().round()
    test_acc[i] = pred[i] == y_test.detach().numpy()

print(f"LOO Acc. = {np.mean(test_acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(y_true,pred)*100:.2f}%")
print(f"LOO F1 score = {f1_score(y_true,pred):.2f}")
cm_display.from_predictions(y_true,pred,normalize=None,display_labels=['Normal','Increase'])
fig, ax = plt.subplots(3,1,figsize=(10,10))
ax[0].plot(train_acc)
ax[0].set_title('Train Accuracy')
ax[1].plot(train_bas)
ax[1].set_title('Train Balanced Accuracy')
ax[2].plot(train_loss)
ax[2].set_title('Train Loss')


# %% Apply SMOTE for imbalanced data (IMPORTANT: Test data including during SMOTE)
X_resampled, y_resampled = SMOTE().fit_resample(feat_lib, y_true)
X_resampled = torch.tensor(X_resampled,dtype=torch.float)
y_resampled = torch.tensor(np.expand_dims(y_resampled,axis = 1),dtype=torch.float)
# LOO
loo = LeaveOneOut()
epochs = 1000
train_acc = []
train_loss = []
train_bas = []
test_acc = [False]*X_resampled.shape[0]
pred = np.zeros((len(test_acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(X_resampled)):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = simple_nn().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
    X_train, y_train = X_resampled[train_idx,:], y_resampled[train_idx]
    X_test, y_test = X_resampled[test_idx,:], y_resampled[test_idx]
    for t in range(epochs):
        acc, loss = train(X_train, y_train, model, loss_fn, optimizer)
        if i == 0:
            train_acc.append(acc)
            train_loss.append(loss)
            model.eval()
            train_pred = model(X_train).detach().numpy().round()
            train_bas.append(bas(train_pred,y_train.reshape(-1).detach().numpy()))
    model.eval()
    pred[i] = model(X_test).detach().numpy().round()
    test_acc[i] = pred[i] == y_test.detach().numpy()

print(f"LOO Acc. = {np.mean(test_acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(y_resampled,pred)*100:.2f}%")
cm_display.from_predictions(y_resampled,pred,normalize=None,display_labels=['Normal','Increase'])
fig, ax = plt.subplots(3,1,figsize=(10,10))
ax[0].plot(train_acc)
ax[0].set_title('Train Accuracy')
ax[1].plot(train_bas)
ax[1].set_title('Train Balanced Accuracy')
ax[2].plot(np.log(train_loss))
ax[2].set_title('Train Loss')
plt.show()

#%% split test data, so SMOTE wont have information on the test data
n_split_test = 5
np.random.seed(0)
inc_data = feat_lib[y_true.reshape(-1)==1,:]
norm_data = feat_lib[y_true.reshape(-1)==0,:]
split_idx = np.random.permutation(inc_data.shape[0])
split_test_inc, split_train_inc = inc_data[split_idx[:n_split_test],:], inc_data[split_idx[n_split_test:],:]
split_idx = np.random.permutation(norm_data.shape[0])
split_test_norm, split_train_norm = norm_data[split_idx[:n_split_test],:], norm_data[split_idx[n_split_test:],:]
split_test_X = np.vstack([split_test_inc, split_test_norm])
split_test_X = torch.tensor(split_test_X, dtype=torch.float)
split_train_X = np.vstack([split_train_inc, split_train_norm])
split_train_y = np.vstack([np.ones((split_train_inc.shape[0],1)), np.zeros((split_train_norm.shape[0],1))])
split_test_y = np.vstack([np.ones((split_test_inc.shape[0],1)), np.zeros((split_test_norm.shape[0],1))])
X_resampled, y_resampled = SMOTE().fit_resample(split_train_X, split_train_y)
X_resampled = torch.tensor(X_resampled,dtype=torch.float)
y_resampled = torch.tensor(np.expand_dims(y_resampled,axis = 1),dtype=torch.float)
# train model
epochs = 500
train_acc = []
train_loss = []
train_bas = []
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
model = simple_nn().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
for t in range(epochs):
    acc, loss = train(X_resampled, y_resampled, model, loss_fn, optimizer)
    train_acc.append(acc)
    train_loss.append(loss)
    model.eval()
    train_pred = model(X_resampled).detach().numpy().round()
    train_bas.append(bas(train_pred,y_resampled.reshape(-1).detach().numpy()))
model.eval()
pred = model(split_test_X).detach().numpy().round()
test_acc = pred == split_test_y
print(f"LOO Acc. = {np.mean(test_acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(split_test_y,pred)*100:.2f}%")
cm_display.from_predictions(split_test_y,pred,normalize=None,display_labels=['Normal','Increase'])
fig, ax = plt.subplots(3,1,figsize=(10,10))
ax[0].plot(train_acc)
ax[0].set_title('Train Accuracy')
ax[1].plot(train_bas)
ax[1].set_title('Train Balanced Accuracy')
ax[2].plot(np.log(train_loss))
ax[2].set_title('Train Loss')
plt.show()

#%% LOO with SMOTE on each training data
# LOO
loo = LeaveOneOut()
epochs = 1000
train_acc = []
train_loss = []
train_bas = []
test_acc = [False]*feat_lib.shape[0]
pred = np.zeros((len(test_acc),1))
for i,(train_idx, test_idx) in enumerate(loo.split(feat_lib)):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = simple_nn().to(device)
    optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)    
    X_train, y_train = feat_lib[train_idx,:], y_true[train_idx]
    X_test, y_test = feat_lib[test_idx,:], y_true[test_idx]
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    y_train = np.expand_dims(y_train,1)
    X_train = torch.tensor(X_train,dtype=torch.float)
    y_train = torch.tensor(y_train,dtype=torch.float)

    for t in range(epochs):
        acc, loss = train(X_train, y_train, model, loss_fn, optimizer)
        if i == 0:
            train_acc.append(acc)
            train_loss.append(loss)
            model.eval()
            train_pred = model(X_train).detach().numpy().round()
            train_bas.append(bas(train_pred,y_train.reshape(-1).detach().numpy()))
    model.eval()
    pred[i] = model(X_test).detach().numpy().round()
    test_acc[i] = pred[i].reshape(-1) == y_test.reshape(-1).detach().numpy()

print(f"LOO Acc. = {np.mean(test_acc)*100:.2f}%")
print(f"LOO Balanced Acc. = {bas(y_true,pred)*100:.2f}%")
print(f"LOO F1 score = {f1_score(y_true,pred):.2f}")
cm_display.from_predictions(y_true,pred,normalize=None,display_labels=['Normal','Increase'])
fig, ax = plt.subplots(3,1,figsize=(10,10))
ax[0].plot(train_acc)
ax[0].set_title('Train Accuracy')
ax[1].plot(train_bas)
ax[1].set_title('Train Balanced Accuracy')
ax[2].plot(np.log(train_loss))
ax[2].set_title('Train Loss')
plt.show()

#%% Find out when to early stop
n_split_test = 5
np.random.seed(1)
inc_data = feat_lib[y_true.reshape(-1)==1,:]
norm_data = feat_lib[y_true.reshape(-1)==0,:]
split_idx = np.random.permutation(inc_data.shape[0])
split_test_inc, split_train_inc = inc_data[split_idx[:n_split_test],:], inc_data[split_idx[n_split_test:],:]
split_idx = np.random.permutation(norm_data.shape[0])
split_test_norm, split_train_norm = norm_data[split_idx[:n_split_test],:], norm_data[split_idx[n_split_test:],:]
split_test_X = np.vstack([split_test_inc, split_test_norm])
split_test_X = torch.tensor(split_test_X, dtype=torch.float)
split_train_X = np.vstack([split_train_inc, split_train_norm])
split_train_y = np.vstack([np.ones((split_train_inc.shape[0],1)), np.zeros((split_train_norm.shape[0],1))])
split_test_y = np.vstack([np.ones((split_test_inc.shape[0],1)), np.zeros((split_test_norm.shape[0],1))])
X_resampled, y_resampled = SMOTE().fit_resample(split_train_X, split_train_y)
X_resampled = torch.tensor(X_resampled,dtype=torch.float)
y_resampled = torch.tensor(np.expand_dims(y_resampled,axis = 1),dtype=torch.float)
# train model
epochs = 1000
train_acc = []
train_loss = []
train_bas = []
test_acc = []
test_bas = []
test_loss = []
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
model = simple_nn().to(device)
optimizer = optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-3)
for t in range(epochs):
    acc, loss = train(X_resampled, y_resampled, model, loss_fn, optimizer)
    train_acc.append(acc)
    train_loss.append(loss)
    model.eval()
    train_pred = model(X_resampled).detach().numpy().round()
    train_bas.append(bas(train_pred,y_resampled.reshape(-1).detach().numpy()))
    test_pred = model(split_test_X)
    pred = test_pred.detach().numpy().round()
    test_acc.append(np.mean(pred == split_test_y))
    test_bas.append(bas(pred,split_test_y))
    test_loss.append(loss_fn(test_pred, torch.tensor(split_test_y,dtype=torch.float)).item())
# print(f"LOO Acc. = {np.mean(test_acc)*100:.2f}%")
# print(f"LOO Balanced Acc. = {bas(split_test_y,pred)*100:.2f}%")
# cm_display.from_predictions(split_test_y,pred,normalize=None,display_labels=['Normal','Increase'])
fig, ax = plt.subplots(3,2,figsize=(10,10))
ax[0][0].plot(train_acc)
ax[0][0].set_title('Train Accuracy')
ax[1][0].plot(train_bas)
ax[1][0].set_title('Train Balanced Accuracy')
ax[2][0].plot(train_loss)
ax[2][0].set_title('Train Loss')
ax[0][1].plot(test_acc)
ax[0][1].set_title('Test Accuracy')
ax[1][1].plot(test_bas)
ax[1][1].set_title('Test Balanced Accuracy')
ax[2][1].plot(test_loss)
ax[2][1].set_title('Test Loss')
plt.show()
print(f"Lowest test loss happens at {np.argmin(test_loss)} epochs.")