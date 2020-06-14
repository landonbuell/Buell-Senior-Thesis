"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import Feature_Utilities as feat_utils

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
        FILE = feat_utils.Assemble_Features(FILE)

        # Extract feature vector from sample
        X = np.append(X,FILE.__getfeatures__())     # add row to design matrix
        del(FILE)                                   # delete file instance

    X = X.reshape(n_samples,-1)     # reshape
    return X                        # return design matrix    
        