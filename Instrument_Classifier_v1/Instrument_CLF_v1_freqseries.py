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

def freqspec_features (wavobj,X,M=(2**12)):
    """
    Produce frequency series features for 'data' attribute on wavobj instance
        Must have already run 'read_raw_wav()' method for this to work!
    --------------------------------
    wavobj (class) : Instance of particular wavfile object
    X (array) : 
    M (int) number of points in array
    --------------------------------
    Returns (N x M) 
    """
    fspace = np.fftfreq(n=M,d=1/44100)      # frequency space axis
    pts = np.where((fspace>=0)&(fspace<=10000))
    xshape = np.shape(X)                    # shape of input array
    Z = fftpack.fft(x=X,n=M,axis=-1)        # compute fft of each row
    Z = np.abs(Z)**2                        # compute power spectrum
    return Z                                # return features

    

