"""
Landon Buell
Instrument Classifier v1
Frequency Series Functions 
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import os

import scipy.signal as signal
import scipy.fftpack as fftpack

import Instrument_CLF_v1_func as func

"""
INSTRUMENT CLASSIFIER V1 - FREQUENCY SERIES FUNCTIONS   

"""

def hanning_window(X,M=(2**12)):
    """
    Apply a hanning window taper to 
    --------------------------------
    X (array) : array to apply Hanning window to each row
    M (int) number of points in array
    --------------------------------
    returns X w/ Hanning taper applied
    """
    taper = signal.hanning(M=M,sym=True)
    Z = np.array([])            # array for output
    for row in X:               # each row
        row = row * taper       # apply hann window
        Z = np.append(Z,row)    # add to Z matrix
    Z = Z.reshape(np.shape(X))
    return Z                    # return the new matrix
