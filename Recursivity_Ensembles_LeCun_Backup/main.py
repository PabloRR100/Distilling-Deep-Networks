#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 09:47:02 2018
@author: pabloruizruiz
"""

## TODO list
'''

 - [~] Track the saturation of every layer --> Plot not very promising ** s
 - [] Track plots of inference on test set at different times of the training:
     - By saving a copy of the model each x epochs or by plotting every x epochs
     
 - [] Track the evolution of the INPUT to the activation functions
     - This is used to track how BatchNorm reduces Internal Covariance Shift
     
'''


import numpy as np
import torch.nn as nn
import torch.optim as optim


''' OPTIMIZER PARAMETERS - Analysis on those '''
BS = 16
LR = 0.01
EPOCHS = 5
MOMEMTUM = 0
WEIGHT_DECAY = 0
NESTEROV = False
BATCHNORM = False


# Choose Dataset
# --------------

dataset = 'moons';  visualize = False
#dataset = 'iris';   visualize = False
#dataset = 'digits'; visualize = False

from data import load_dataset
X, y = load_dataset(name=dataset, visualize=visualize)


# Split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#df_train = to_df(X_train, y_train); #df_test = to_df(X_test, y_test)
#scatterplot([df, df_train, df_test], ['Original data', 'Training set', 'Test set'])

from data import create_torch_dataset
tr_loader = create_torch_dataset(X_train, y_train, BS, shuffle=True)
ts_loader = create_torch_dataset(X_test, y_test, BS=BS, shuffle=True)


# Config Neural Network
# ---------------------

n_layers = 4                                                # Number of hidden Layers
lay_size = 50                                               # Width of hidden layers
inp_dim = X_train.shape[1]                                  # Input dimension
n_class = len(np.unique(y_train))                           # Output dimension


# Define Training
# --------------    

from utils import timeit
from train_valid_test import train_epoch, valid_epoch

#@timeit
def train(model, criterion, optimizer, results, EPOCHS, LR):
    for epoch in range(EPOCHS):    
        
        # Training
        model.train()
        lss, acc, preds = train_epoch(model, tr_loader, criterion, optimizer, LR, results)
        print('Epoch {} -- Training: Loss: {}, Accy: {}'.format(epoch, lss, acc))
        if epoch % 10 == 0: print('Preds: ', preds)
        
        # Validation
        model.eval()
        
        lss, acc = valid_epoch(model, ts_loader, criterion, results)    
        if epoch % 10 == 0: print('Epoch {} -- Validation: Loss: {}, Accy: {}'.format(epoch, lss, acc))
    

''' Test model '''
class TestNet(nn.Module):
    
    def __init__(self):
        super(TestNet, self).__init__()
        self.name = "TestNet"
        self.fc1 = nn.Linear(2,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
    def collect_stats(self,x):
        pass
        
    
from utils import Results
results = [Results()]
testNet = TestNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(testNet.parameters(), LR)
train(testNet, criterion, optimizer, results[0], EPOCHS, LR)
models = [testNet]
# torch.save(model, model.name + 'model.pkl')



################################################################
        # Run Experiments
        
################################################################

models = []

from utils import Results
from models.fcnn import TorchNet
criterion = nn.CrossEntropyLoss()

#results = dict(train_loss = [], train_accy = [], valid_loss = [], valid_accy = [])
results = []

# Create variants of different models to compare
modelN = TorchNet('No Activation', inp_dim, n_class, lay_size, n_layers, actf='none', track_stats=True, recursive=0)
modelS = TorchNet('Sigmoid', inp_dim, n_class, lay_size, n_layers, actf='sigm', track_stats=True, recursive=0)
modelT = TorchNet('TanH', inp_dim, n_class, lay_size, n_layers, actf='tanh', track_stats=True, recursive=0)
modelR = TorchNet('ReLU', inp_dim, n_class, lay_size, n_layers, actf='relu', track_stats=True, recursive=0)

# Select models to train
#models += [modelN, modelS, modelT, modelR]
models = [modelR]
for model in models:
    
    r = Results()
    optimizer = optim.SGD(model.parameters(), LR, MOMEMTUM, WEIGHT_DECAY, nesterov=NESTEROV)
    model_no_recursive_params = [model, criterion, optimizer, r]
    train_no_recursive_params = [EPOCHS, LR]
    train(*model_no_recursive_params, *train_no_recursive_params)
    results.append(r)




# ANALYIS
#####################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Performance Analysis
# ---------------------

from plots import true_vs_pred
confusion_matrices = true_vs_pred(models, X_train, X_test, y_test)
colors = ['red', 'blue', 'green', 'grey', 'black', 'yellow', 'purple']

#import numpy as np
#import torch
#from torch.autograd import Variable
#X_fake = np.random.randn(X_test.shape[0], X_test.shape[1])
#X_fake = torch.tensor(X_fake, dtype=torch.long)
#X_fake.type()
#
#model = models[0]
#
#y_fake = model(Variable(X_fake))


# Train Validation Loss Accuracy
m = models[0]
def lab(a,m):
    if a == 'T': return 'Training {}'.format(m.name)
    else: return 'Validation {}'.format(m.name)

fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15,15))
c = colors[:len(models)]
for i, r in enumerate(results):
    sns.lineplot(range(EPOCHS), r.train_loss, color=c[i], label=lab('T',models[i]), ax=axs[0])
    sns.lineplot(range(EPOCHS), r.valid_loss, color=c[i], dashes=True, label=lab('V',models[i]), ax=axs[0])
    sns.lineplot(range(EPOCHS), r.train_accy, color=c[i], label=lab('T',models[i]), ax=axs[1])
    sns.lineplot(range(EPOCHS), r.valid_accy, color=c[i], dashes=True, label=lab('V',models[i]), ax=axs[1])
for i in range(len(axs[0].lines)):
    if i % 2 != 0:
        axs[0].lines[i].set_linestyle("--")
        axs[1].lines[i].set_linestyle("--")
axs[0].set_title('Loss')
axs[1].set_title('Accuracy')
fig.suptitle('Network --> Act f: {}.    # Hidden Layers: {}.    Hidden Layers Width: {}' \
             .format(m.activation, m.n_lay, m.lay_size))
plt.plot()



model = models[0]
X_fake = np.random.rand()


# Network Analysis
# -----------------

net = models[0]

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
from plots import distribution_of_graphs
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

