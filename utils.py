
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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
    fig, axs = plt.subplots(ncols=2, figsize=(15,15))
    sns.scatterplot(x='X1', y='X2', hue='y', data=df_test, 
                    legend=False, palette=sns.color_palette("Set1", n_colors=2),
                    ax=axs[0]).set_title('Real Distribution')
    sns.scatterplot(x='X1', y='X2', hue='y', data=df_pred, 
                    legend=False, palette=sns.color_palette("Set2", n_colors=2),
                    ax=axs[1]).set_title('Predicted Distribution')
    plt.title('Prediction results')
    plt.plot()
    

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


def ratioweights(net, learning_rate):
    '''
    A rough heuristic is that this ratio should be somewhere around 1e-3. 
    If it is lower than this then the learning rate might be too low. 
    If it is higher then the learning rate is likely too high.
    '''
    W1_mean = float(net.fc1.weight.data.mean())
    W1_var  = float(net.fc1.weight.data.var())
    W2_mean = float(net.fc2.weight.data.mean())
    W2_var  = float(net.fc2.weight.data.var())
    
    dW1 = net.fc1.weight.grad.data.numpy()
    dW2 = net.fc2.weight.grad.data.numpy()
    
    W1_scale = np.linalg.norm(net.fc1.weight.grad)
    update1 = -learning_rate * dW1 
    update_scale1 = np.linalg.norm(update1.ravel())
    ratio1 = float(update_scale1 / W1_scale)
    
    W2_scale = np.linalg.norm(net.fc2.weight.grad)
    update2 = -learning_rate * dW2
    update_scale2 = np.linalg.norm(update2.ravel())
    ratio2 = float(update_scale2 / W2_scale)
    return W1_mean, W1_var, W2_mean, W2_var, ratio1, ratio2
    