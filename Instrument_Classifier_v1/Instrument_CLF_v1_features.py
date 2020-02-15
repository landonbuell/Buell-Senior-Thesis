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
    - Calls to 'timeseries' and 'freqseries' to produce matrices (X by convention)
    - Each row of each X matrix is a particular sample case
    - Each column of each X matrix is a particular feature

"""

def train_wavfile (wavfile,clf_dict,classes):
    """
    Train single wav file on classifiers 
    --------------------------------
    wavfile (inst) : instance of .wav file to train on classifiers
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    classes (array) : array of class labels
    --------------------------------
    Returns classifier dictionary w/ trained models
    """   
    
    X,y = timeseries.waveform_features(wavfile,M=2**12)     # get waveform features
    setattr(wavfile,'data',X.flatten())                     # flatten, set attrb
    clf_dict['time_clf'].partial_fit(X,y,classes=classes)   # partial fit data set

    #hann = freqseries.hanning_window(X,M=2**12)             # Hann window to waveform
    #X,y = freqseries.freqspec_features(wavfile,hann)        # get FFT features
    #clf_dict['freq_clf'].partial_fit(X,y,classes=classes)   # partial fit data set

    return clf_dict

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