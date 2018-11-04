#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''

- [] Extend possibility to automatically adjust the network to input config
    - Allowing to track network capacities

 - [~] Timeit one epoch pass scratch/torch sgd/momentum
 - [X] Compute hidden values sizes
 - [] Track the saturation of every layer
 - [] Track plots of inference on test set at different times of the training:
     - By saving a copy of the model each x epochs or by plotting every x epochs
     
 - [] Track the evolution of the INPUT to the activation functions
     - This is used to track how BatchNorm reduces Internal Covariance Shift
'''


import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split



# DATASETS
# --------

from utils import to_df, scatterplot

X, y = make_moons(n_samples=5000, random_state=42, noise=0.1)
df = to_df(X, y)
df.head()

#scatterplot([df])


# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

df_train = to_df(X_train, y_train)
df_test = to_df(X_test, y_test)

#scatterplot([df, df_train, df_test], ['Original data', 'Training set', 'Test set'])



# NEURAL NETWORK
# --------------

inp_dim = 2
lay_size = 100
learning_rate = 0.1
n_class = len(np.unique(y_train))

EPOCHS = 50
BATCHSIZE = 64


pytorch = dict(train_loss = list(), train_accy = list(), 
               valid_loss = list(), valid_accy = list())


# Training PyTorch Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


# Pass data to PyTorch Dataloader ## PyTorch does not use onehotencoded
from utils import create_torch_dataset
    
tr_loader = create_torch_dataset(X_train, y_train, BATCHSIZE, shuffle=True)
ts_loader = create_torch_dataset(X_test, y_test, BS=BATCHSIZE, shuffle=False)

    
from networks import TorchNet
torchnet = TorchNet(inp_dim, n_class, lay_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(torchnet.parameters(), learning_rate, momentum=0, weight_decay=0)


torchnet.train()
# Training (and validating)
for epoch in range(EPOCHS):    
            
    # Run minibaches from the training dataset
    for i, (X, labels) in enumerate(tr_loader):
        
        X, labels = Variable(X), Variable(labels)
        
        # Forward pass
        torchnet.zero_grad()
        y_pred = torchnet(X)
        s, preds = torch.max(y_pred.data, 1)
        
        # Compute loss 
        loss = criterion(y_pred, labels)            
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Collect stats 
        torchnet.collect_stats(learning_rate)

    # Compute and store epoch results
    correct, total = 0, 0
    total += y_pred.size(0)
    correct += int(sum(preds == labels)) 
    accuracy = correct / total
    
    lss = round(loss.item(), 3)
    acc = round(accuracy * 100, 2)

    pytorch['train_loss'].append(lss)
    pytorch['train_accy'].append(acc)    
    
    print('Epoch {} -- Training: Loss: {}, Accy: {}'
          .format(epoch, lss, acc))
    
    # Validation
    for i, (X, labels) in enumerate(ts_loader):
        
        X, labels = Variable(X), Variable(labels)
        
        # Forward pass
        y_pred = torchnet(X)
        s, preds = torch.max(y_pred.data, 1)
        
        # Compute loss 
        loss = criterion(y_pred, labels)           
        
    # Compute and store epoch results
    correct, total = 0, 0
    total += y_pred.size(0)
    correct += int(sum(preds == labels)) 
    accuracy = correct / total
    
    lss = round(loss.item(), 3)
    acc = round(accuracy * 100, 2)

    pytorch['valid_loss'].append(lss)
    pytorch['valid_accy'].append(acc)
        
    print('Epoch {} -- Validation: Loss: {}, Accy: {}'
          .format(epoch, lss, acc))



# Plot Predictions
from utils import true_vs_pred
y_pred_all = torchnet(Variable(torch.tensor(X_test, dtype=torch.float32)))
y_pred_all = torch.max(y_pred_all.data, 1)[1].detach().numpy()

df_test = to_df(X_test, y_test)
df_pred = to_df(X_test, y_pred_all)

true_vs_pred(df_test, df_pred)



# PyTorch Network Analysis
# ------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

net = torchnet

W1_stats = net.weight_stats['W1']
W2_stats = net.weight_stats['W2']
W1_scale = net.weight_stats['rW1']
W2_scale = net.weight_stats['rW2']


## The Var is way bigger than the Mean !!??
stast = pd.DataFrame(W1_stats)
plt.figure(figsize=(15,15))
plt.plot(range(len(stast['mean'])), stast['mean'])
plt.fill_between(range(len(stast['mean'])), 
                 stast['mean'] + stast['var'], 
                 stast['mean'] - stast['var'], 
                 alpha=0.2, label='W1 mean')
plt.plot()



# Mean and var of the weights separately
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
xaxis = range(len(W1_stats['mean']))
sns.lineplot(xaxis, W1_stats['mean'], label='W1 mean', ax=axs[0,0])
sns.lineplot(xaxis, W1_stats['var'], label='W1 var', ax=axs[0,1])
sns.lineplot(xaxis, W2_stats['mean'], label='W2 mean', ax=axs[1,0])
sns.lineplot(xaxis, W2_stats['var'], label='W2 var', ax=axs[1,1])
plt.plot()


# Evolution of the gradients
from utils import normalize_gradients
xaxis = range(len(W1_stats['mean']))
norm_dW1, norm_dW2 = normalize_gradients(
        net.weight_stats['gradW1'], net.weight_stats['gradW2'], type='standard')
plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
sns.lineplot(xaxis, net.weight_stats['gradW1'], ax=ax1, color='blue').set_title('grad W1')
sns.lineplot(xaxis, net.weight_stats['gradW2'], ax=ax2, color='red').set_title('grad W2')
sns.lineplot(xaxis, net.weight_stats['gradW1'], ax=ax3, color='blue', label='grad W1')
sns.lineplot(xaxis, net.weight_stats['gradW2'], ax=ax3, color='red', label='grad W2')
sns.kdeplot(norm_dW1, shade=True, ax=ax4)
sns.kdeplot(norm_dW2, shade=True, ax=ax4)
plt.plot()


# Histogram of weights gradients
plt.figure(figsize=(15,15))
sns.kdeplot(norm_dW1, shade=True, ax=ax4)
sns.kdeplot(norm_dW2, shade=True, ax=ax4)
plt.plot()


# Ratio weight / updata (should be around 1e-3)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
sns.lineplot(xaxis, W1_scale, label='W1 ratio', ax=axs[0])
sns.lineplot(xaxis, W2_scale, label='W2 ratio', ax=axs[1])
plt.plot()


# Saturation of the layers
downsampling = 40
axis = range(0, len(net.L1['mean']), downsampling)

plt.figure(figsize=(15,15))
plt.title('Activation value (mean and variance)')
plt.plot(range(len(net.L1['mean'])), net.L1['mean'], color='red')
plt.errorbar(axis, net.L1['mean'][::downsampling], net.L1['var'][::downsampling], linestyle='None', color='red')
plt.show()

