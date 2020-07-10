"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import time

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils
import Math_Utilities as math_utils
import Neural_Network_Models


def target_array (fileobjs,n_classes,matrix=True):
    """
    Construct target matrix or target vector
    --------------------------------
    fileobjs (inst) : File Object instances with 'target' attribute
    n_classes (int) : Number of unique classification classes
    matrix (bool) : If true (default), target array is one-hot-encoded matrix
    --------------------------------
    Return matrix/vector of target arr
    """
    y = np.array([x.target for x in fileobjs])      # target attrb
    if matrix == True:                              # if matrix...
        y = keras.utils.to_categorical(y,n_classes) # one-hot-enc
    return y                                        # return targets

    
      