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
import Instrument_CLF_v1_MLfunc as MLfunc

"""
INSTRUMENT CLASSIFIER V1 - FEATURE PRODUCTION
    This program contains functions and methods that will be used to oversee
    the generation of features for each wavfile class instance
    - Calls to 'timeseries' and 'freqseries' to produce rows for X matrix
    - Each row of each X matrix is a particular sample case
    - Each column of each X matrix is a particular feature

"""

def timeseries_features(wavfile):
    """
    Use features from each wavfile instance
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    --------------------------------
    Returns array of time series features for classification
    """   
    attack = timeseries.attack_frac(wavfile,start=0.1,stop=0.9) 
    release = timeseries.release_frac(wavfile,start=0.1,stop=0.7)
    srt_to_max,max_to_stp = timeseries.max_amp(wavfile,ref=0.1)

    # assemble into a vector
    feature_array = np.array([attack,release,srt_to_max,max_to_stp],
                             dtype=float)
    return feature_array 

def freqseries_features(wavfile):
    """
    Use features from each wavfile instance
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    --------------------------------
    Returns array of freqency series features for classification
    """
    timeseries.reshape_waveform(wavfile)    # make npts % M = 0


    # assemble into vector
    feature_array = np.array([],
                             dtype=float)
    return feature_array

def concatenate_features(wavfile):
    """
    Concatenate all features into  array for single sample instance
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    --------------------------------
    Returns array of (1 x N) containing all features 
    """
    sample_row = np.array([timeseries_features(wavfile)],
                          dtype=float)
    sample_row = sample_row.ravel()         # flatten array
    return sample_row                   # return the row
