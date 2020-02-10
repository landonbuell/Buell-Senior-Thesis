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

def time_domain_features(training_wavfiles,clf,test=False):
    """
    Extract features and train 
    --------------------------------
    training_wavfiles (list) : : List of all instances of .wav file objects
    clf (obj) : Classifier Object to fit w/ training data
    --------------------------------

    """    
    for wav in training_wavfiles:       # for each bit of training data:    
        X,y = timeseries.waveform_features(wav) # produce features and labels
        data = X.flatten()              # flatten the time array
        setattr(wav,'data',data)        # overwrite waveform attrb
        if test == False:               # if not testinf data
            clf.partial_fit(X,y)        # fit the data set
        else:
            pass
        


        