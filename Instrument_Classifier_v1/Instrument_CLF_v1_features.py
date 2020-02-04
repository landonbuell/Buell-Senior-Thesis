"""
Landon Buell
Instrument Classifier v1
Machine Learning Functions
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import os

import Instrument_CLF_v1_func as func
import Instrument_CLF_v1_timeseries as timeseries
import Instrument_CLF_v1_freqseries as freqseries

"""
INSTRUMENT CLASSIFIER V1 - FEATURE PRODUCTION
    This program contains functions and methods that will be used to oversee
    the generation of features for each wavfile class instance
    - Calls to 'timeseries' and 'freqseries' to produce matrices (X by convention)
    - Each row of each X matrix is a particular sample case
    - Each column of each X matrix is a particular feature

"""

def waveform_features (wavobj,M=2**12):
    """
    Produce time series features for 'data' attribute on wavobj instance
        Must have already run "read_raw_wav() method for this to work!
    --------------------------------
    wavobj (class) : Instance of particular wavfile object
    M (int) : number of features per file obj (recc. 2^N w/ N and int)
    --------------------------------
    Returns (N x M) array of features amd (M x 1) array of labels
    """
    ext = len(wavobj.data) % M          # remaining idx left over
    X = wavobj.data[:ext]               # 