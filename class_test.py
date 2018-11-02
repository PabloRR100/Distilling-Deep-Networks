
import numpy as np
from sklearn.utils import shuffle

class Network():
    '''
    Network for binary classification with:
        1 hidden layer, softmax output and cross-entropy loss function.
    '''
    
    def __init__(self, in_f, out_c, lay_size=100, learning_rate=0.001):
        super(Network, self).__init__()
        
        self.inp_dim = in_f
        self.classes = out_c
        
        self.W1 = np.random.rand(in_f, lay_size)
        self.W2 = np.random.rand(lay_size, out_c)
        
        
class SGD_Optimizer():
    ''' SGD minibatch '''
    
    def __init__(self, Network, n_iter):
        super(SGD_Optimizer, self).__init__()
        self.net = Network
        self.iters = n_iter
    


inp_dim = 2
n_class = 10

EPOCHS = 10
BATCHSIZE = 50
ITERATIONS = 10

accs = np.zeros(EPOCHS)

net = Network(inp_dim, n_class)
optimizer = SGD_Optimizer(net, EPOCHS)


## Working
optimizer.net.W1 = None
net.W1 == None # true
