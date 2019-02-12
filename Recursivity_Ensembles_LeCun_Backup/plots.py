
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.metrics import confusion_matrix


import torch
from torch.autograd import Variable

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
    

def true_vs_pred(models, X_train, X_test, y_test):
    
    confusion_matrices = []
    fig, axs = plt.subplots(ncols=2, nrows=len(models), figsize=(15,15), constrained_layout=True)
    for m, model in enumerate(models):
    
        y_pred_all = model(Variable(torch.tensor(X_test, dtype=torch.float32)))
        y_pred_all = torch.max(y_pred_all.data, 1)[1].numpy()
        confusion_matrices.append(confusion_matrix(y_pred_all, y_test))
        
        # Targets vs Predictions (Only for 2D Inputs)
        if X_train.shape[1] == 2:
            
            df_test = to_df(X_test, y_test)
            df_pred = to_df(X_test, y_pred_all)
            
            s = len(df_pred['y'].unique())
            
            if len(models) > 1:
                sns.scatterplot(x='X1', y='X2', hue='y', data=df_test, 
                                legend=False, palette=sns.color_palette("Set1", n_colors=2),
                                ax=axs[m][0]).set_title('Real Distribution {}'.format(model.name))
                sns.scatterplot(x='X1', y='X2', hue='y', data=df_pred, 
                                legend=False, palette=sns.color_palette("Set2", n_colors=s),
                                ax=axs[m][1]).set_title('Predicted Distribution {}'.format(model.name))
            else:
                sns.scatterplot(x='X1', y='X2', hue='y', data=df_test, 
                                legend=False, palette=sns.color_palette("Set1", n_colors=2),
                                ax=axs[0]).set_title('Real Distribution {}'.format(model.name))
                sns.scatterplot(x='X1', y='X2', hue='y', data=df_pred, 
                                legend=False, palette=sns.color_palette("Set2", n_colors=s),
                                ax=axs[1]).set_title('Predicted Distribution {}'.format(model.name))
            plt.suptitle('Prediction results')
            plt.plot()
    return confusion_matrices

    
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

