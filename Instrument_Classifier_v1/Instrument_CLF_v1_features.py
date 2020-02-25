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
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    classes (array) : array of class labels
    --------------------------------
    Returns array of time series features for classification
    """   
    attack = timeseries.attack_frac(wavfile,start=0.1,stop=0.9) 
    release = timeseries.release_frac(wavfile,start=0.1,stop=0.7)

    # assemble into a vector
    feature_array = np.array([attack,release])
    return feature_array 

def freqseries_features(wavfile):
    """
    Use features from each wavfile instance
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    classes (array) : array of class labels
    --------------------------------
    Returns array of freqency series features for classification
    """
    n_peaks = None

def test_wavfile (wavfile,clf_dict,classes):
    """
    Test single wav file on classifiers 
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    classes (array) : array of class labels
    --------------------------------
    Returns prediction on wavfile
    """  
    actual = wavfile.class_num                              # actual class
    X,y = timeseries.waveform_features(wavfile,M=2**12)     # get waveform features
    time_pred = MLfunc.prediction_function(clf_dict['time_clf'],X)  # predict case X
    return actual,time_pred