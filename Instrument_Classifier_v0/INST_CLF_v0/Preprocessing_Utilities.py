"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import tensorflow.keras as keras
import os
import sys

from sklearn.preprocessing import StandardScalar

            #### FUNCTION DEFINITIONS ####

def construct_targets (fileobjs,matrix=True):
    """
    Construct target array object 
    --------------------------------
    fileobjs (iter) : List of object instances w/ 'target' attribute
    matrix (bool) : If true, targets are one-hot-encoded into matrix,
        else a vector is returned
    --------------------------------
    Return target object and nmber of unique classes
    """
    y = np.array([x.target for x in fileobjs])  # use target attribute
    n_classes = len(np.unique(y))               # number of classes
    if matrix == true:                          # if one-hot-enc
        y = keras.utils.to_categorical(y,n_classes)
    return y,n_classes                          # return target & classes

def Scale_X (X):
    """
    Apply standard preprocessing scaling to design matrix, X
    --------------------------------
    X (arr) : Standard design matrix, shape (n_samples x n_features)
    --------------------------------
    Return scaled design matrix
    """
    scalar = StandardScalar()   # scalar instance
    scalar = scalar.fit(X)      # fit frame
    X_new = scalar.transform(X) # tranform
    return X_new                # return new design matrix

