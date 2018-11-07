#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''

 - [] Extend to be able to pass network config and build it directly
 - [] Timeit one epoch pass scratch/torch sgd/momentum
 - [X] Compute hidden values sizes
 - [] Track the saturation of every layer
 - [] Check if the reason of no learning could have to do with the zero_grad()
         and if the heavy change of the var of the weights could be a good indicator
         
 - [] Report how big LR yields to exploding/vanishing gradient seen as [nan, nan]
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

EPOCHS = 50
BATCHSIZE = 64


# Placeholders for results
pure_python = dict(train_loss = list(), train_accy = list(), 
                   valid_loss = list(), valid_accy = list())


# Networks

from networks import Network, SGD_Optimizer

net = Network(inp_dim, n_class, lay_size, learning_rate)
optimizer = SGD_Optimizer(net, EPOCHS)


# Training (and validating)
for epoch in range(EPOCHS):
    
    ## reset network ??
    
    
    # Training
    l, a = optimizer.minibatch_SGD(X_train, y_train, BATCHSIZE)
    pure_python['train_loss'].append(l)
    pure_python['train_accy'].append(a)        
    
    # Validation
    y_pred = list()
    current_loss = list()
    correct, total = 0, 0
    for i, (x,c) in enumerate(zip(X_test, y_test)):
        
        # One hot encoded to calculate the loss
        y_true = np.zeros(n_class)
        y_true[int(c)] = 1.
        _, prob = net.forward(x)
        current_loss.append(net.crossentropy(prob, y_test[i]))
        
        # Accuracy
        y = np.argmax(prob)
        y_pred.append(y)
        
        total += 1
        if y_pred[i] == y_test[i]: 
            correct += 1
    
    # Calculate loss and accy
    pure_python['valid_loss'].append(np.mean(current_loss))
    pure_python['valid_accy'].append(correct / total * 100)
    
    print('Epoch: {}, Loss: {}, Accy: {}'.format(epoch, l, a))





# Results
# -------

mode = pure_python ## Choose pure_python / pytorch to see results

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
plt.title('Loss Function')
sns.lineplot(range(EPOCHS), mode['train_loss'], label='Trainining')
sns.lineplot(range(EPOCHS), mode['valid_loss'], label='Validation')
plt.plot()


plt.figure()
plt.title('Accuracy Evolution')
sns.lineplot(range(EPOCHS), mode['train_accy'], label='Training')
sns.lineplot(range(EPOCHS), mode['valid_accy'], label='Validation')
plt.plot()


# Predictions 

from utils import true_vs_pred
_, y_pred = net.forward(X_test)
y_pred_all = np.zeros(len(y_pred))

for i,r in enumerate(y_pred):
    if y_pred[i,0] == r.max():
        y_pred_all[i] = 0
    else:
        y_pred_all[i] = 1

df_test = to_df(X_test, y_test)
df_pred = to_df(X_test, y_pred_all)

true_vs_pred(df_test, df_pred)





# PyTorch Network Analysis
# ------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

net = net

W1_stats = net.weight_stats['W1']
W2_stats = net.weight_stats['W2']
W1_scale = net.weight_stats['rW1']
W2_scale = net.weight_stats['rW2']


## The Var is way bigger than the Mean !!??
stats1 = pd.DataFrame(W1_stats)
stats2 = pd.DataFrame(W2_stats)
plt.figure(figsize=(15,15))
plt.plot(range(len(stats1['mean'])), stats1['mean'])
plt.fill_between(range(len(stats1['mean'])), 
                 stats1['mean'] + stats1['var'], 
                 stats1['mean'] - stats1['var'], 
                 alpha=0.2, label='W1 mean')
plt.plot(range(len(stats2['mean'])), stats2['mean'])
plt.fill_between(range(len(stats2['mean'])), 
                 stats2['mean'] + stats2['var'], 
                 stats2['mean'] - stats2['var'], 
                 alpha=0.2, label='W2 mean')
plt.plot()



# Mean and var of the weights separately
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,15))
xaxis = range(len(W1_stats['mean']))
sns.lineplot(xaxis, W1_stats['mean'], label='W1 mean', ax=axs[0,0])
sns.lineplot(xaxis, W1_stats['var'], label='W1 var', ax=axs[0,1])
sns.lineplot(xaxis, W2_stats['mean'], label='W2 mean', ax=axs[1,0])
sns.lineplot(xaxis, W2_stats['var'], label='W2 var', ax=axs[1,1])
plt.plot()


# Evolution and Histogram of the gradients
from utils import normalize_gradients
norm_dW1, norm_dW2 = normalize_gradients(
        net.weight_stats['gradW1'], net.weight_stats['gradW2'], type='standard')
plt.figure(figsize=(15,15))
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=1)
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=1)
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=2)
sns.lineplot(xaxis, net.weight_stats['gradW1'], ax=ax1, color='blue').set_title('grad W1')
sns.lineplot(xaxis, net.weight_stats['gradW2'], ax=ax2, color='red').set_title('grad W2')
sns.lineplot(xaxis, net.weight_stats['gradW1'], ax=ax3, color='blue', alpha=0.5, label='grad W1')
sns.lineplot(xaxis, net.weight_stats['gradW2'], ax=ax3, color='red', alpha=0.5, label='grad W2')
sns.kdeplot(norm_dW1, shade=True, ax=ax4)
sns.kdeplot(norm_dW2, shade=True, ax=ax4)
plt.plot()


# Ratio weight / updata (should be around 1e-3)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,15))
sns.lineplot(xaxis, W1_scale, label='W1 ratio', ax=axs[0])
sns.lineplot(xaxis, W2_scale, label='W2 ratio', ax=axs[1])
plt.plot()


# Saturation of the layers
downsampling = 2000
axis = range(0, len(net.L1['mean']), downsampling)

plt.figure(figsize=(15,15))
plt.title('Activation value (mean and variance)')
plt.plot(axis, net.L1['mean'][::downsampling], color='blue')
plt.errorbar(axis, net.L1['mean'][::downsampling], net.L1['var'][::downsampling], linestyle='None', color='red')
plt.show()









