"""
Landon Buell
PHYS 799
Instrument Classifier v0
17 June 2020
"""

            #### IMPORTS ####
              
import numpy as np
from sklearn.preprocessing import StandardScaler

            #### FUNCTION DEFINITIONS ####

def RMS_Energy (X):
    """
    Compute RMS energy of object X along last axis
        Output contains 1 less dimension
    --------------------------------
    X (arr) : Array-like of floats 
    --------------------------------
    Return RMS of array X
    """           
    RMS = np.sqrt(np.mean(X**2,axis=-1))# compute mean
    return RMS                          # return 

def Scale_X (X):
    """
    Apply standard preprocessing scaling to design matrix, X
    --------------------------------
    X (arr) : Standard design matrix, shape (n_samples x n_features)
    --------------------------------
    Return scaled design matrix
    """
    scaler = StandardScaler()   # scalar instance
    scaler = scaler.fit(X)      # fit frame
    X_new = scaler.transform(X) # transform
    return X_new                # return new design matrix

