"""
Landon Buell
Instrument Classifier v1
Time Series Functions 
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import os

import scipy.signal as signal

import Instrument_CLF_v1_func as func

"""
INSTRUMENT CLASSIFIER V1 - FREQUENCY SERIES FUNCTIONS   
    - Creating time axes
    - organizing waveform

"""


def time_axis (N,rate=44100):
    """
    create a time axis for tiem series data
    --------------------------------
    N (int) : number of points in data
    rate (int) : number of samples per second (44100 by default)
    --------------------------------
    return numpy array for time axis
    """
    return np.arange(N)/rate

def waveform_features (wavobj,M=(2**12)):
    """
    Produce time series features for 'data' attribute on wavobj instance
        Must have already run 'read_raw_wav()' method for this to work!
    --------------------------------
    wavobj (class) : Instance of particular wavfile object
    M (int) : number of features per file obj (recc. 2^N w/ N as int)
    --------------------------------
    Returns (N x M) array of features amd (M x 1) array of labels
    """
    ext = len(wavobj.data) % M          # remaining idx left over
    X = wavobj.data[ext:]               # crop waveform
    N = int(len(X)/M)                   # n rows
    X = X.reshape(N,M)                  # reshape 
    y = np.ones((N,1))*wavobj.class_num # target labels
    y = y.ravel()                       # flatten 
    return X,y                          # return matrix & targets


