#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''
 - [~] Timeit one epoch pass scratch/torch sgd/momentum
 - [X] Compute hidden values sizes
 - [] Track the saturation of every layer
 - [] Track plots of inference on test set at different times of the training:
     - By saving a copy of the model each x epochs or by plotting every x epochs
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
learning_rate = 0.001
n_class = len(np.unique(y_train))

EPOCHS = 10
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
for epoch in range(20):    
            
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

net = torchnet

W1_stats = net.W1['W1_stats']
W2_stats = net.['W2_stats']
W1_scale = torch_stats['W1_scale']
W2_scale = torch_stats['W2_scale']

import seaborn as sns
import matplotlib.pyplot as plt

# Mean of the weights
plt.figure()
sns.lineplot(range(len(W1_stats)), [i[0] for i in W1_stats], label='W1 mean')
sns.lineplot(range(len(W1_stats)), [i[0] for i in W2_stats], label='W2 mean')  ## W2 is not changing !!
plt.plot()

# Variance of the weight
plt.figure()
sns.lineplot(range(len(W1_stats)), [i[1] for i in W1_stats], label='W1 variance')
sns.lineplot(range(len(W1_stats)), [i[1] for i in W2_stats], label='W2 variance')
plt.plot()

# Ratio weight / updata (should be around 1e-3)
plt.figure()
sns.lineplot(range(len(W1_stats)), W1_scale, label='W1 ratio')
sns.lineplot(range(len(W1_stats)), W2_scale, label='W2 ratio')
plt.plot()

