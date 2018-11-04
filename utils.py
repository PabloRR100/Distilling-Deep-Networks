
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



def normalize_gradients(a,b, type:str):
    
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

    a = list(chain(*list(scale(np.array(a).reshape(-1,1)))))
    b = list(chain(*list(scale(np.array(b).reshape(-1,1)))))
    return a, b
