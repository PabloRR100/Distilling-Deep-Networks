#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''

- [X] Extend possibility to automatically adjust the network to input config
    - Allowing to track network capacities
- [] Extend all the plots to that extension

 - [~] Timeit one epoch pass scratch/torch sgd/momentum
 - [X] Compute hidden values sizes
 - [~] Track the saturation of every layer --> Plot not very promising ** s
 - [] Track plots of inference on test set at different times of the training:
     - By saving a copy of the model each x epochs or by plotting every x epochs
     
 - [] Track the evolution of the INPUT to the activation functions
     - This is used to track how BatchNorm reduces Internal Covariance Shift
     
'''

import pickle
import numpy as np
from sklearn.datasets import make_moons, make_classification, load_iris, load_digits
from sklearn.model_selection import train_test_split


# DATASETS
# --------

from utils import to_df, scatterplot

## Make moon dataset
X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
df = to_df(X, y)


## Iris dataset
#iris = load_iris()
#X = iris.data[:, :]
#y = iris.target


## NIST Dataset -- 8x8 digits 1000 images
#nist = load_digits()
#X = nist.data
#y = nist.target


# Make random classification dataset
#X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, 
#                    n_repeated=0, n_classes=2, n_clusters_per_class=2, random_state=42)


#scatterplot([df])


# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#df_train = to_df(X_train, y_train)
#df_test = to_df(X_test, y_test)
#scatterplot([df, df_train, df_test], ['Original data', 'Training set', 'Test set'])



# NEURAL NETWORK CONFIG
# ---------------------

inp_dim = X_train.shape[1]
n_layers = 2
lay_size = 10
n_class = len(np.unique(y_train))

EPOCHS = 100
BATCHSIZE = 16
learning_rate = 0.01
momentum = 0
weight_decay = 0
nesterov = False

results = dict(train_loss = list(), train_accy = list(), 
               valid_loss = list(), valid_accy = list())


# Training PyTorch Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# Define Data Loaders
# --------------------
## PyTorch does not use onehotencoded
from utils import create_torch_dataset
    
tr_loader = create_torch_dataset(X_train, y_train, BATCHSIZE, shuffle=True)
ts_loader = create_torch_dataset(X_test, y_test, BS=BATCHSIZE, shuffle=True)


# Define Network
# --------------    

from models.fcnn import TorchNet
from train_valid_test import train_epoch, valid_epoch

torchnet = TorchNet(inp_dim, n_class, lay_size, n_layers, track_stats=False, recursive=2)


# Define Training
# ---------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(torchnet.parameters(), learning_rate, momentum, weight_decay, nesterov=nesterov)

for epoch in range(EPOCHS):    
    
    # Training
    torchnet.train()
    lss, acc = train_epoch(torchnet, tr_loader, criterion, optimizer, learning_rate, results)
    print('Epoch {} -- Training: Loss: {}, Accy: {}'.format(epoch, lss, acc))
    
    # Validation
    torchnet.eval()
    
    lss, acc = valid_epoch(torchnet, ts_loader, criterion)    
    print('Epoch {} -- Validation: Loss: {}, Accy: {}'.format(epoch, lss, acc))


# torch.save(torchnet, 'torchnet.pkl')


# ANALYIS
#####################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Performance Analysis
# ---------------------

y_pred_all = torchnet(Variable(torch.tensor(X_test, dtype=torch.float32)))
y_pred_all = torch.max(y_pred_all.data, 1)[1].detach().numpy()
    
# Targets vs Predictions (Only for 2D Inputs)
if X_train.shape[1] == 2:
    
    from utils import true_vs_pred
    df_test = to_df(X_test, y_test)
    df_pred = to_df(X_test, y_pred_all)
    
    true_vs_pred(df_test, df_pred)

# Confusion Matrix
confusion_matrix(y_test, y_pred_all)


# Train Validation Loss Accuracy
fig, axs = plt.subplots(nrows=2, ncols=1)
sns.lineplot(range(EPOCHS), results['train_loss'], label='Training', ax=axs[0])
sns.lineplot(range(EPOCHS), results['valid_loss'], label='Validation', ax=axs[0])
sns.lineplot(range(EPOCHS), results['train_accy'], label='Training', ax=axs[1])
sns.lineplot(range(EPOCHS), results['valid_accy'], label='Validation', ax=axs[1])
axs[0].set_title('Loss')
axs[1].set_title('Accuracy')
plt.plot()



# Network Analysis
# -----------------

net = torchnet

Winp_stats = net.weight_stats['Winp']
Whid_stats = net.weight_stats['Whid']
Wout_stats = net.weight_stats['Wout']
Winp_scale = net.weight_stats['rWinp']
Whid_scale = net.weight_stats['rWhid']
Wout_scale = net.weight_stats['rWout']

xaxis = range(len(Winp_stats['mean']))


## What does it mean that the variance is bigger than the mean?
## What should I plot, mean +- std or +- var given that std values are > 0 < 1 ??
ls = net.n_lay
fig, axs = plt.subplots(nrows=2+ls, ncols=1)
stast = pd.DataFrame(Winp_stats)
axs[0].plot(range(len(stast['mean'])), stast['mean'])
axs[0].fill_between(range(len(stast['mean'])), stast['mean'] + stast['var']**2, 
                 stast['mean'] - stast['var']**2, alpha=0.2, label='W1 mean')

for l in range(ls):
    stast = pd.DataFrame(Whid_stats[l])
    axs[l+1].plot(range(len(stast['mean'])), stast['mean'])
    axs[l+1].fill_between(range(len(stast['mean'])), stast['mean'] + stast['var']**2, 
                     stast['mean'] - stast['var']**2, alpha=0.2, label='W1 mean')

stast = pd.DataFrame(Wout_stats)
axs[ls+1].plot(range(len(stast['mean'])), stast['mean'])
axs[ls+1].fill_between(range(len(stast['mean'])), stast['mean'] + stast['var']**2, 
                 stast['mean'] - stast['var']**2, alpha=0.2, label='W1 mean')
plt.plot()



# Mean and var of the weights separately
ls = net.n_lay
fig, axs = plt.subplots(nrows=2+ls, ncols=2)
sns.lineplot(xaxis, Winp_stats['mean'], label='WInp mean', ax=axs[0,0])
sns.lineplot(xaxis, Winp_stats['var'], label='WInp var', ax=axs[0,1])
for l in range(ls):
    sns.lineplot(xaxis, Whid_stats[l]['mean'], label='W{} mean'.format(l+1), ax=axs[l+1,0])
    sns.lineplot(xaxis, Whid_stats[l]['var'], label='W{} var'.format(l+1), ax=axs[l+1,1])    
sns.lineplot(xaxis, Wout_stats['mean'], label='WOut mean', ax=axs[ls+1,0])
sns.lineplot(xaxis, Wout_stats['var'], label='WOut var', ax=axs[ls+1,1])
plt.plot()


# Evolution of the gradients
from utils import extract_dictgrads
normgrads = extract_dictgrads(net)
xaxis = range(len(Winp_stats['mean']))
plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
sns.lineplot(xaxis, net.weight_stats['gradWinp'], ax=ax1, color='blue').set_title('grad W1')
sns.lineplot(xaxis, net.weight_stats['gradWout'], ax=ax2, color='red').set_title('grad W2')
sns.lineplot(xaxis, net.weight_stats['gradWinp'], ax=ax3, color='blue', label='grad W1')
sns.lineplot(xaxis, net.weight_stats['gradWout'], ax=ax3, color='red', label='grad W2')
[sns.kdeplot(normgrads[i], shade=True, clip=(-1.5, 1.5), ax=ax4) for i in normgrads.keys()]
plt.plot()


# Histogram of gradients
from utils import distribution_of_graphs
distribution_of_graphs(net)


# Ratio weight / updata (should be around 1e-3)
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,15))
sns.lineplot(xaxis, Winp_scale, label='W1 ratio', ax=axs[0])
sns.lineplot(xaxis, Whid_scale[1], label='W1 ratio', ax=axs[1])
sns.lineplot(xaxis, Wout_scale, label='W2 ratio', ax=axs[2])
plt.plot()


# Saturation of the layers
downsampling = 40
axis = range(0, len(net.lInp['mean']), downsampling)

plt.figure(figsize=(15,15))
plt.title('Activation value (mean and variance)')
plt.plot(range(len(net.lInp['mean'])), net.lInp['mean'], color='red')
plt.errorbar(axis, net.lInp['mean'][::downsampling], net.lInp['var'][::downsampling], linestyle='None', color='red')
for l in range(net.n_lay):
    plt.plot(range(len(net.lHid[l]['mean'])), net.lHid[l]['mean'])
    plt.errorbar(axis, net.lHid[l]['mean'][::downsampling], net.lHid['var'][l][::downsampling], linestyle='None')
plt.show()

