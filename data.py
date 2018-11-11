# -*- coding: utf-8 -*-

import pickle
import numpy as np
from utils import to_df, scatterplot

from sklearn.datasets import make_moons, make_classification, load_iris, load_digits


def load_dataset(name:str, visualize=False, *args):
    
    options = ['moons', 'irirs', 'digits']
    assert name in options, 'Please, choose a valid dataset from {}'.format(options)
    
    if name == 'moons':
        ## Make moon dataset
        X, y = make_moons(n_samples=10000, random_state=42, noise=0.1)
        if visualize:
            df = to_df(X, y)
            scatterplot([df])

        
    if name == 'iris':
        # Iris datasetr[]
        iris = load_iris()
        X = iris.data[:, :]
        y = iris.target
    
    if name == 'digits':
    ## NIST Dataset -- 8x8 digits 1000 images
        nist = load_digits()
        X = nist.data
        y = nist.target
        
    return X,y

    

