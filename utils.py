

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


