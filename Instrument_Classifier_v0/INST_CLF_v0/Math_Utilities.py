"""
Landon Buell
PHYS 799
Instrument Classifier v0
17 June 2020
"""

            #### IMPORTS ####
              
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

"""
Math_Utilities.py - "Math Utilities"
    Contains Definitions for general mathematical functions
"""

            #### FUNCTION DEFINITIONS ####

def Distribution_Data (X):
    """
    Analyze properties of an array of FP values
    --------------------------------
    X (arr) : Array of FP number to analyze as distribution
    --------------------------------
    Return array of [mean,median,mode,variance]
    """
    X = X.ravel()           # flatten
    mean = np.mean(X)       # avg
    median = np.median(X)   # median
    mode,cnts = stats.mode(X)    # mode
    var = np.var(X)         # variance
    return np.array([mean,median,mode[-1],var])

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

