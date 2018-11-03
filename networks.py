

import numpy as np
from sklearn.utils import shuffle
from torch import nn

''' 

Dimensionality
==============

# Inputs/Outputs are column vectors
X = (BS, inp) = (2, 1)
y = (out, 1)
y_one_hot = y* = (out, 1)

W1 (inp, hid) - (2,100) 
W2 (hid, out) - (100,2)

h = f(W1.T * X) = (hid, inp) * (inp, 1) = (hid, 1)
o = W2.T * h    = (out, hid) * (hid, 1) = (out, 1) 
p = softmax(o)  = (out,1)

L = SUM(y*log(p)) = (1)

dL/do = p - y* = (out, 1)
dL/dW2 = (do/dW2) * (dL/do).T = h * (p - y*).T = (hid, 1) * (1, out) = (hid, out)

dL/dh  = dL/do * dO/dh = (out, 1) = W2*(p - y*) = (hid, out) * (out, 1) = (hid, 1)
dL/dW1 = dL/dh * dh/dW = X * (W2(p - y*)).T = (inp, 1) * (1, hid) = (inp, hid)

W1 = W1 + lr * dW1
W2 = W2 + lr * dW2

'''

'''

Tracking the Network
====================

Update ratio of weights
-----------------------
A rough heuristic is that this ratio should be somewhere around 1e-3. 
If it is lower than this then the learning rate might be too low. 
If it is higher then the learning rate is likely too high.

'''


class TorchNet(nn.Module):
    ''' Same network using PyTorch '''
    
    def __init__(self, in_f, out_c, lay_size=100, print_sizes=False):
        super(TorchNet, self).__init__()
        
        self.fc1 = nn.Linear(in_f, lay_size)
        self.fc2 = nn.Linear(lay_size, out_c)
        self.relu = nn.ReLU(inplace=True)
        self.p = print_sizes     
        
        self.weight_stats = dict(
                self.W1 = dict(mean = list(), var = list()),
                self.W2 = dict(mean = list(), var = list()),
                self.rW1 = list(),
                self.rW2 = list(),
                )
        
        self.L1 = dict(mean = list(), var = list())
        
        
#    @profile
    def forward(self, x):
         
        # Layer 1
        if self.p: print("\t FC1 input size: ", x.size())        
        x = self.relu(self.fc1(x))
        self.L1['mean'].append(float(x.mean()))
        self.L1['var'].append(float(x.mean()))
        
        # Layer 2
        if self.p: print('\t FC2 input size: ', x.size())
        x = self.fc2(x)
        
        if self.p: print("\t Output size: ", x.size())
        return x
    
    def collect_stats(self, lr):
        
        self.W1['mean'].append(float(self.fc1.weight.data.mean()))
        self.W1['var'].append(float(self.fc1.weight.data.var()))
        self.W2['mean'].append(float(self.fc2.weight.data.mean()))
        self.W2['var'].append(float(self.fc2.weight.data.var()))
        
        dW1 = self.fc1.weight.grad.numpy()
        dW2 = self.fc2.weight.grad.numpy()
        
        W1_scale = np.linalg.norm(self.fc1.weight.grad)
        update1 = -lr * dW1 
        update_scale1 = np.linalg.norm(update1.ravel())
        self.rW1.append(float(update_scale1 / W1_scale))
        
        W2_scale = np.linalg.norm(self.fc2.weight.grad)
        update2 = -lr * dW2
        update_scale2 = np.linalg.norm(update2.ravel())
        self.rW2.append(float(update_scale2 / W2_scale))
 

class Network():
    '''
    Network for binary classification with:
        1 hidden layer, softmax output and cross-entropy loss function.
    '''
    
    def __init__(self, in_f, out_c, lay_size=100, learning_rate=1e-4):
        super(Network, self).__init__()
        
        self.inp_dim = in_f
        self.classes = out_c
        
        self.W1 = np.random.rand(in_f, lay_size)
        self.W2 = np.random.rand(lay_size, out_c)
        
        self.lr = learning_rate
        
        self.W1_stats = list()
        self.W2_stats = list()
        self.W1_scale = list()
        self.W2_scale = list()
        
    def ratioweights(self, W, update):
        param_scale = np.linalg.norm(W.ravel())
        update_scale = np.linalg.norm(update.ravel())
        return update_scale / param_scale
        
        
    def relu(self, x):
        x[x < 0] = 0
        return x
        
    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))
  
    def crossentropy(self, p, y):
        return -np.sum(y * np.log(p+1e-9)) / p.shape[0]
      
    def forward(self, x):
        a = x @ self.W1
        h = self.relu(a)
        z = h @ self.W2
        p = self.softmax(z)
        return h, p 
      
    def backward(self, xs, hs, grads):
        dW2 = hs.T @ grads          # dL / dW2
        dh = grads @ self.W2.T      # dL / dh
        dh[hs <= 0] = 0             # dh / dh (through relu)
        dW1 = xs.T @ dh             # dL / dW1
        return dW1, dW2

    def update_weights(self, dW1, dW2):
        self.W1 = self.W1 + self.lr * dW1
        self.W2 = self.W2 + self.lr * dW2
        
        print('dW2: ', dW2)
        
        self.W1_stats.append((self.W1.mean(), self.W1.var()))
        self.W2_stats.append((self.W2.mean(), self.W2.var()))
        self.W1_scale.append(self.ratioweights(self.W1, self.lr * dW1))
        self.W2_scale.append(self.ratioweights(self.W2, self.lr * dW2))
       
        
        
class SGD_Optimizer():
    ''' SGD minibatch '''
    
    def __init__(self, Network, n_iter):
        super(SGD_Optimizer, self).__init__()
        self.net = Network
        self.iters = n_iter
        self.train_loss = []
        self.train_accy = []
        
        self.lr_reducer = True
        
        
    def get_batch(self, X, y, i, BS):
        return X[i:i + BS], y[i:i + BS]
        
        
    def minibatch_SGD(self, X, y, BS):
        
        X, y = shuffle(X, y) # See them in another order
        
        # Run minibaches from the training dataset
        for i in range(0, X.shape[0], BS):
            
            # For every single pair (X,y) on that chunk
            X_mini, y_mini = self.get_batch(X, y, i, BS)
            
            # Compute the gradient of that minibatch
            dW1, dW2 = self.backward_pass(X_mini, y_mini)
            
            # Update the weight after processed the minibatch
            self.net.update_weights(dW1, dW2)
                
        ep_train_loss = np.mean(np.array(self.train_loss))
        ep_train_accy = np.mean(np.array(self.train_accy))
        return ep_train_loss, ep_train_accy
    
        
    def backward_pass(self, X, y):
        ''' Acumulate minibatch info into 1 single pass '''
        xs, hs, grads = self.calculate_gradient(X, y)
        return self.net.backward(xs, hs, grads)

        
    def calculate_gradient(self, X_train, y_train):
        ''' Wrapper of minibatch statistics into single arrays '''
        xs, hs, grads = [], [], []
    
        for x, cls_idx in zip(X_train, y_train):
            
            h, y_pred = self.net.forward(x)
    
            # Create true label (1-hot-encoded)
            y_true = np.zeros(self.net.classes)
            y_true[int(cls_idx)] = 1.
    
            # Compute the gradient of output layer
            loss = self.net.crossentropy(y_pred, y_true)
            #grad = (y_true - y_pred) / len(y_pred)
            grad = (y_true - y_pred)
#            print('Grad: ', grad)
            self.train_loss.append(loss)
    
            # Accumulate the informations of minibatch
            xs.append(x)
            hs.append(h)
            grads.append(grad)
            
            # Calculate training loss and accuracy
            self.train_accy.append((y_pred == y_true).sum() / y_true.size)
    
        # Backprop using the informations we get from the current minibatch
        return np.array(xs), np.array(hs), np.array(grads)
    
    
        