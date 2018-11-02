#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

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
learning_rate = 0.01
n_class = len(np.unique(y_train))

EPOCHS = 10
BATCHSIZE = 64

train_loss = list()
train_accy = list()
valid_loss = list()
valid_accy = list()


from networks import Network, SGD_Optimizer, TorchNet

#net = Network(inp_dim, n_class, lay_size, learning_rate)
#optimizer = SGD_Optimizer(net, EPOCHS)
#
#
## Training (and validating)
#for epoch in range(EPOCHS):
#    
#    ## reset network ??
#    
#    
#    # Training
#    l, a = optimizer.minibatch_SGD(X_train, y_train, BATCHSIZE)
#    train_loss.append(l)
#    train_accy.append(a)
#    
#    if epoch % 5 == 0:
#        print(net.W1[:2,:2])
#        
#    
#    # Validation
#    y_pred = list()
#    current_loss = list()
#    for i, (x,c) in enumerate(zip(X_test, y_test)):
#        
#        # One hot encoded to calculate the loss
#        y_true = np.zeros(n_class)
#        y_true[int(c)] = 1.
#        _, prob = net.forward(x)
#        current_loss.append(net.crossentropy(prob, y_test[i]))
#        
#        # Accuracy
#        y = np.argmax(prob)
#        y_pred.append(y)
#    
#    # Calculate loss and accy
#    valid_loss.append(np.mean(current_loss))
#    valid_accy.append((y_pred == y_test).sum() / y_test.size)
#    
#    if epoch % 5 == 0:
#        print('Epoch: {}, Loss: {}, Accy: {}'.format(epoch, l, a))



# Training PyTorch Model
import torch.optim as optim
from torch.autograd import Variable

torchnet = TorchNet(inp_dim, n_class, lay_size)
optimizer = optim.SGD(torchnet.parameters(), learning_rate, momentum=0, weight_decay=0)


def get_batch(self, X, y, i, BS):
        return Variable(X[i:i + BS]), Variable(y[i:i + BS])

# Training (and validating)
for epoch in range(EPOCHS):    
    
    # Get batches
    X, y = shuffle(X, y) # See them in another order
        
        # Run minibaches from the training dataset
        for i in range(0, X.shape[0], BS):
            
            # For every single pair (X,y) on that chunk
            X_mini, y_mini = self.get_batch(X, y, i, BS)
    
    
    
    
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
    valid_loss.append(np.mean(current_loss))
    valid_accy.append((y_pred == y_test).sum() / y_test.size)
    
    if epoch % 5 == 0:
        print('Epoch: {}, Loss: {}, Accy: {}'.format(epoch, l, a))





# Results
# -------
    
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure()
sns.lineplot(range(EPOCHS), train_loss)
sns.lineplot(range(EPOCHS), valid_loss)
plt.plot()


plt.figure()
sns.lineplot(range(EPOCHS), train_accy)
sns.lineplot(range(EPOCHS), valid_accy)
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





