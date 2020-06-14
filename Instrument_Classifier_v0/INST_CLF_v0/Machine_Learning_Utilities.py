"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import Timespace_Features as time_feats
import Freqspace_Features as freq_feats

            #### FUNCTION DEFINITIONS ####

def Assemble_Features (FILE):
    """
    Create & Collect all classification features
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return (1 x N) array of features
    """
    FILE = time_feats.rise_decay_time(FILE) # rise & decay 
    
    return FILE                    # return file with feature vector

def Design_Matrix (FILE_OBJECTS):
    """
    Construct Standard Machine-Learning Design Matrix
        (n_samples x n_features)
    --------------------------------
    FILE_OBJECTS (iter) : Iterable of all file objects to use
    --------------------------------
    Return design matrix, X
    """
    X = np.array([])                # empty design matrix
    n_samples = len(FILE_OBJECTS)   # number of file samples

    for I,FILE in enumerate(FILE_OBJECTS):  # iterate through samples

        FILE = FILE.read_audio()            # read .WAV file
        
        # Assign features to object instance
        FILE = Assemble_Features(FILE)

        # Extract feature vector from sample
        X = np.append(X,FILE.__getfeatures__())     # add row to design matrix

    X = X.reshape(n_samples,-1)     # reshape
    return X                        # return design matrix    
        