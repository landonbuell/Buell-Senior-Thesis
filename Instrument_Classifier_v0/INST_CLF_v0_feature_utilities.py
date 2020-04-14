"""
Landon Buell
Instrument Classifier v0
Feature Extraction
6 April 2020
"""

            #### IMPORTS ####

import numpy as np

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_time_utilities as time_utils
import INST_CLF_v0_freq_utilities as freq_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils

def timeseries (wavfile):
    """
    Collect all training features for audio file based on
        Time spectrum data 
    --------------------------------
    wavfile (inst) : Instance of .wav file w/ waveform attribute
    --------------------------------
    Return array of time series features
    """
    features = np.array([])
    features = np.append(features,time_utils.rise_decay_time(wavfile.waveform))
    return features

def freqseries (wavfile):
    """
    Collect all training features for audio file based on
        Frequency spectrum data 
    --------------------------------
    wavfile (inst) : Instance of .wav file w/ waveform attribute
    --------------------------------
    Return array of frequency series features
    """
    features = np.array([])
    

    
    return features