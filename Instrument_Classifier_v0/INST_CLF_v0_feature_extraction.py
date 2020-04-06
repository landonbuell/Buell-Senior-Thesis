"""
Landon Buell
Instrument Classifier v0
Feature Extraction
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils

def Xy_matrices(wavfile_objects):
    """
    Construct feature matrix "X" (n_samples x n_features) and
        target vector "y" (n_samples x 1)
    --------------------------------
    wavfile_objects (list) : List of wavfile object instances to read
    -------------------------------R-
    Return  -feature matrix "X" (n_samples x n_features) 
            -target vector "y" (n_samples x 1)
    """
    n_samples = len(wavfile_objects)        # number of samples
    y = np.array([x.instrument for x in wavfile_objects])   # target vector
    ENCODE_DICTIONARY = ML_utils.target_label_encoder(y)
    y = np.array([ENCODE_DICTIONARY[x] for x in y])
    X = np.array([])                        # feature matrix
    
    return X,y