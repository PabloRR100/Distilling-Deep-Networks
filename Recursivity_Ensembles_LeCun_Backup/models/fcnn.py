
import sys
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


Tracking the Network
====================

Update ratio of weights
-----------------------
A rough heuristic is that this ratio should be somewhere around 1e-3. 
If it is lower than this then the learning rate might be too low. 
If it is higher then the learning rate is likely too high.


Assumptions
===========

- All the layers are the same width
- All the layes share the same activation function

'''


class TorchNet(nn.Module):
    ''' Same network using PyTorch '''
    
    
    def __init__(self, name:str, inp:int, out:int, hid:int,  n_layers:int, 
                 actf: str = 'relu', recursive=None, track_stats=True, print_sizes=False):
        super(TorchNet, self).__init__()
        
        opt = ['none', 'relu', 'sigm', 'tanh']
        err = 'Select a correct activation function from: {}'.format(opt)
        assert actf in opt, err
        
        self.name = name
        self.lay_size = hid
        self.n_lay = n_layers
        self.activation = actf
        self.fcInp = nn.Linear(inp, hid, bias=False)
        
        self.fcHid = nn.ModuleList([nn.Linear(hid, hid, bias=False) for _ in range(self.n_lay)])        
        self.fcOut = nn.Linear(hid, out, bias=False)
     
        self.track_stats = track_stats
        self.recursive = None if recursive == 0 else recursive
        
        if actf == 'none': self.actf = self.linact
        if actf == 'relu': self.actf = nn.ReLU(inplace=True)
        if actf == 'sigm': self.actf = nn.Sigmoid()
        if actf == 'tanh': self.actf = nn.Tanh()
        
        if self.track_stats:
            self.weight_stats = dict(
                    # Weight stats
                    Winp = dict(vals = list(), mean = list(), var = list()),
                    Whid = [dict(vals = list(), mean = list(), var = list()) for _ in range(self.n_lay)],
                    Wout = dict(vals = list(), mean = list(), var = list()),
                    # Grad stats
                    gradWinp = list(),
                    gradWhid = [list() for _ in range(self.n_lay)],
                    gradWout = list(),
                    # Update stats
                    rWinp = list(),
                    rWhid = [list() for _ in range(self.n_lay)],
                    rWout = list(),
                    )
            
            self.lInp = dict(mean = list(), var = list())
            self.lHid = [dict(mean = list(), var = list()) for _ in range(self.n_lay)]        
        
    def linact(self, x):
        return x
        
    def forward(self, x):
         
        # Input Layer         
        x = self.actf(self.fcInp(x))
        if self.track_stats:
            self.lInp['mean'].append(float(x.mean()))
            self.lInp['var'].append(float(x.mean()))
        
        # Hidden Layers
        for l in range(self.n_lay):
            
            x = self.actf(self.fcHid[l](x))
            if self.track_stats:
                self.lHid[l]['mean'].append(float(x.mean()))
                self.lHid[l]['mean'].append(float(x.var()))
            
            # Recursive Layer (last layer)
            if l == max(range(self.n_lay)) and self.recursive is not None:
                for _ in range(self.recursive):
                    x = self.actf(self.fcHid[l](x))

        # Output Layer 
        x = self.fcOut(x)
        return x
    
    
    def collect_stats(self, lr):
        
        if self.track_stats:
            ## TODO: rethink weather is better to stor the vals or mean and var for plotting
            # Weight Values Means and Variances 
            self.weight_stats['Winp']['vals'].append(self.fcInp.weight.data.numpy())
            self.weight_stats['Winp']['mean'].append(float(self.fcInp.weight.data.mean()))
            self.weight_stats['Winp']['var'].append(float(self.fcInp.weight.data.var()))
            
            for i in range(self.n_lay):
                self.weight_stats['Whid'][i]['vals'].append(self.fcHid[i].weight.data.numpy())
                self.weight_stats['Whid'][i]['mean'].append(float(self.fcHid[i].weight.data.mean()))
                self.weight_stats['Whid'][i]['var'].append(float(self.fcHid[i].weight.data.var()))
            
            self.weight_stats['Wout']['vals'].append(self.fcOut.weight.data.numpy())
            self.weight_stats['Wout']['mean'].append(float(self.fcOut.weight.data.mean()))
            self.weight_stats['Wout']['var'].append(float(self.fcOut.weight.data.var()))
            
            ## TODO - Think of networks utils to encapsulate these methods
            # Gradients
            dWinp = self.fcInp.weight.grad.numpy()
            dWhid = [self.fcHid[i].weight.grad.numpy() for i in range(self.n_lay)]
            dWout = self.fcOut.weight.grad.numpy()
            
            # Update values
            Winp_scale = np.linalg.norm(dWinp)
            self.weight_stats['gradWinp'].append(Winp_scale)
            updateInp = -lr * dWinp 
            update_scaleInp = np.linalg.norm(updateInp.ravel())
            self.weight_stats['rWinp'].append(float(update_scaleInp / Winp_scale))
            
            ## TODO - Look if numpy could allow to rewrite most of the list comprehension lines
            for i in range(self.n_lay):        
                Whid_scale = np.linalg.norm(dWhid[i])
                self.weight_stats['gradWhid'][i].append(Whid_scale)
                updateHid = -lr * dWhid[i]
                update_scaleHid = np.linalg.norm(updateHid.ravel())
                self.weight_stats['rWhid'][i].append(float(update_scaleHid / Whid_scale))
            
            Wout_scale = np.linalg.norm(dWout)
            self.weight_stats['gradWout'].append(Wout_scale)
            updateOut = -lr * dWout
            update_scaleOut = np.linalg.norm(updateOut.ravel())
            self.weight_stats['rWout'].append(float(update_scaleOut / Wout_scale))
 
    

##############################################################################
##############################################################################
        


class Network():
    '''
    Network for binary classification with:
        1 hidden layer, softmax output and cross-entropy loss function.
    '''
    
    def __init__(self, in_f, out_c, lay_size=100, 
                 learning_rate=1e-4, print_sizes=False):
        super(Network, self).__init__()
        
        self.inp_dim = in_f
        self.classes = out_c
        
        self.W1 = np.random.rand(in_f, lay_size)
        self.W2 = np.random.rand(lay_size, out_c)
        
        self.lr = learning_rate
        self.p = print_sizes     
        
        self.weight_stats = dict(
                W1 = dict(vals = list(), mean = list(), var = list()),
                W2 = dict(vals = list(), mean = list(), var = list()),
                gradW1 = list(),
                gradW2 = list(),
                rW1 = list(),
                rW2 = list(),
                )
        
        self.L1 = dict(mean = list(), var = list())
        
    def ratioweights(self, W, update):
        param_scale = np.linalg.norm(W.ravel())
        update_scale = np.linalg.norm(update.ravel())
        return update_scale / param_scale
    
    def checknans(self, x):
        return np.isnan(x).any()
    
    def sigmoid(self, z):   
        return 1 / (1+np.exp(-z))

    def dsigmoid(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

    def tanh(self, z):
        return np.tanh(z)
    
    def dtanh(self, z):
        return (1 - np.tanh(z)^2)
        
    def relu(self, x):
        x[x < 0] = 0
        return x
    
    def drelu(self, x, h):
        x[h <= 0] = 0
        return x
        
    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))
  
    def crossentropy(self, p, y):
        return -np.sum(y * np.log(p+1e-9)) / p.shape[0]
      
    def collect_stats(self, dW1, dW2):
        
        self.weight_stats['W1']['vals'].append(self.W1)
        self.weight_stats['W1']['mean'].append(float(self.W1.mean()))
        self.weight_stats['W1']['var'].append(float(self.W2.var()))
        
        self.weight_stats['W2']['vals'].append(self.W2)
        self.weight_stats['W2']['mean'].append(float(self.W2.mean()))
        self.weight_stats['W2']['var'].append(float(self.W2.var()))
                
        Winp_scale = np.linalg.norm(dW1)
        self.weight_stats['gradW1'].append(Winp_scale)
        updateInp = -self.lr * dW1 
        update_scaleInp = np.linalg.norm(updateInp.ravel())
        self.weight_stats['rW1'].append(float(update_scaleInp / Winp_scale))
        
        Wout_scale = np.linalg.norm(dW2)
        self.weight_stats['gradW2'].append(Wout_scale)
        updateOut = -self.lr * dW2
        update_scaleOut = np.linalg.norm(updateOut.ravel())
        self.weight_stats['rW2'].append(float(update_scaleOut / Wout_scale))    
    
    def forward(self, x):
        if self.p: print("\t FC1 input size: ", x.size())        
        # Layer 1
        a = x @ self.W1
        h = self.sigmoid(a)
#        h = self.tanh(a)
#        h = self.relu(a)
        self.L1['mean'].append(float(h.mean()))
        self.L1['var'].append(float(h.mean()))
        
        # Layer 2
        if self.p: print('\t FC2 input size: ', x.size())
        z = h @ self.W2
        if self.p: print("\t Output size: ", x.size())
        p = self.softmax(z)
        return h, p 
      
    def backward(self, xs, hs, grads):
        dW2 = hs.T @ grads          # dL / dW2
        dh = grads @ self.W2.T      # dL / dh
        
        dh = self.dsigmoid(dh)      # dh / dh (through sigmoid)
#        dh = self.dtanh(dh)         # dh / dh (through sigmoid)
#        dh = self.drelu(dh, hs)     # dh / dh (through relu)
#        
        dW1 = xs.T @ dh             # dL / dW1
        
        explode_gradient = True if self.checknans(dW1) or self.checknans(dW2) else False
        #assert not explode_gradient, 'Vanishing/Exploding Gradient !'
        if explode_gradient:
            print('Vanishing/Exploding Gradient !')
            print('Last gradient: \n dW1: {} \n dW2: {}'.format(dW1[:2], dW2[:2]))
            sys.exit(0)
        return dW1, dW2

    def update_weights(self, dW1, dW2):
        
        self.collect_stats(dW1, dW2)
        
        self.W1 = self.W1 + self.lr * dW1
        self.W2 = self.W2 + self.lr * dW2
        
 
####################################
        
        
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
        ep_train_accy = np.mean(np.array(self.train_accy)) * 100
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
            y_pred_ = np.zeros(len(y_pred))
            if y_pred[0] == y_pred.max():
                y_pred_[0] = 1
            else:
                y_pred_[1] = 1
            self.train_accy.append((y_pred_ == y_true).sum() / y_true.size)
    
        # Backprop using the informations we get from the current minibatch
        return np.array(xs), np.array(hs), np.array(grads)
    
    
        