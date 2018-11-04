#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''
 - [] Timeit one epoch pass scratch/torch sgd/momentum
 - [X] Compute hidden values sizes
 - [] Track the saturation of every layer
 - [] Check if the reason of no learning could have to do with the zero_grad()
         and if the heavy change of the var of the weights could be a good indicator
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

scatterplot([df, df_train, df_test], ['Original data', 'Training set', 'Test set'])



# NEURAL NETWORK
# --------------

inp_dim = 2
lay_size = 100
learning_rate = 0.001
n_class = len(np.unique(y_train))

EPOCHS = 100
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
    
    if epoch % 5 == 0:
        print(net.W1[:2,:2])
        
    
    # Validation
    y_pred = list()
    current_loss = list()
    for i, (x,c) in enumerate(zip(X_test, y_test)):
        
        # One hot encoded to calculate the loss
        y_true = np.zeros(n_class)
        y_true[int(c)] = 1.
        _, prob = net.forward(x)
        current_loss.append(net.crossentropy(prob, y_test[i]))
        
        # Accuracy
        y = np.argmax(prob)
        y_pred.append(y)
    
    # Calculate loss and accy
    pure_python['valid_loss'].append(np.mean(current_loss))
    pure_python['valid_accy'].append((y_pred == y_test).sum() / y_test.size)
    
    if epoch % 5 == 0:
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
plt.figure('Accuracy Evolution')
sns.lineplot(range(EPOCHS), mode['train_accy'], label='Training')
sns.lineplot(range(EPOCHS), mode['valid_accy'], label='Validation')
plt.plot()



# Network Analysis
# ----------------

W1_stats = net.W1_stats
W2_stats = net.W2_stats
W1_scale = net.W1_scale
W2_scale = net.W2_scale


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




