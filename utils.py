
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

# GENERAL USE
# -----------

class Results():
    ''' Object to store training / validation results'''
    def __init__(self):
        super(Results, self).__init__()
        self.train_loss = [] 
        self.train_accy = []
        self.valid_loss = []
        self.valid_accy = []


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


# PLOTTING
# --------

def to_df(X, y):
    return pd.concat((pd.DataFrame(X, columns=['X1', 'X2']), 
                pd.DataFrame(y, columns=['y'])), axis=1)

def scatterplot(dfs:list, titles:list = [None]):
    
    assert len(dfs) == len(titles), 'List must be same lenght'
    if len(dfs) > 1: 
        fig, axs = plt.subplots(ncols=len(dfs), figsize=(15,15))
        for i in range(len(dfs)):
            sns.scatterplot(x='X1', y='X2', hue='y', data=dfs[i], legend=False,
                            palette=sns.color_palette("Set1", n_colors=2), 
                            ax=axs[i]).set_title(titles[i])
    else: 
        plt.figure(figsize=(15,15))
        sns.scatterplot(x='X1', y='X2', hue='y', data=dfs[0], legend=False,
                        palette=sns.color_palette("Set1", n_colors=2))
    plt.show()
    

def true_vs_pred(df_test, df_pred):
    # If the network is outputing all to one single class:
    s = len(df_pred['y'].unique())
    
    fig, axs = plt.subplots(ncols=2, figsize=(15,15))
    sns.scatterplot(x='X1', y='X2', hue='y', data=df_test, 
                    legend=False, palette=sns.color_palette("Set1", n_colors=2),
                    ax=axs[0]).set_title('Real Distribution')
    sns.scatterplot(x='X1', y='X2', hue='y', data=df_pred, 
                    legend=False, palette=sns.color_palette("Set2", n_colors=s),
                    ax=axs[1]).set_title('Predicted Distribution')
    plt.title('Prediction results')
    plt.plot()
    
    
def distribution_of_graphs(net):
    
    dictgrads = extract_dictgrads(net)
    df = pd.melt(pd.DataFrame(dictgrads).iloc[::-1])
    df.columns = ['grad', 'x']
    
    pal = sns.cubehelix_palette(len(df['grad'].unique()), rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="grad", hue="grad", aspect=15, height=5, palette=pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=0.6, lw=1.5, bw=.2)
    g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2) ## White contour
    g.map(plt.axhline, y=0, lw=2, clip_on=False) ## Will serve as the x axis
    
    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="bottom", transform=ax.transAxes)
#        ax.set_xlim([-1.5, 1.5])
    g.map(label, "x")
    
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.75)
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    return g
 

def onehotencode(vec, n_class):
    #tr_labels = onehotencode(y_train, n_class)
    #ts_labels = onehotencode(y_test, n_class)
    placeh = np.zeros((len(vec), n_class))
    for i in range(len(vec)):
        placeh[i, vec[i]] = 1.
    return placeh


import torch
import torch.utils.data as data_utils
def create_torch_dataset(inputs, labels, BS, shuffle):
    t = data_utils.TensorDataset(
            torch.tensor(inputs, dtype=torch.float32),     ## Inputs are float
            torch.tensor(labels, dtype=torch.torch.int64)) ## Labels are int
    loader = data_utils.DataLoader(t, batch_size=BS, shuffle=shuffle)
    return loader


def extract_dictgrads(net):
    
    grads = list()
    grads.append(net.weight_stats['gradWinp'])
    [grads.append(net.weight_stats['gradWhid'][l]) for l in range(net.n_lay)]
    grads.append(net.weight_stats['gradWout'])
    normgrads = normalize_gradients(grads, type='standard')
    
    dictgrads = OrderedDict()
    dictgrads['dW Inp'] = normgrads[0]
    for i in range(1,len(normgrads)):
        dictgrads['dW Hid {}'.format(i)] = normgrads[i]
    dictgrads['dW Out'] = normgrads[-1]
    return dictgrads
    


def normalize_gradients(vs:list, type:str):
    
    options = ['standard', 'normal']
    err = 'Choose between valid scaling ["standard" / "normal"]'
    assert type in options, err
    
    from itertools import chain
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import StandardScaler

    def normalize(data):
        scaler = MinMaxScaler(feature_range=(-1,1))
        return scaler.fit_transform(data)
    
    def standarize(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    scale = standarize if type == 'standard' else normalize
    
    ns = list()
    for v in vs:
        ns.append(list(chain(*list(scale(np.array(v).reshape(-1,1))))))
    return ns


# Count parameters of a model 
def count_parameters(model):
    ''' Count the parameters of a model '''
    return sum(p.numel() for p in model.parameters())