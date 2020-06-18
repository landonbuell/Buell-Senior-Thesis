"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import tensorflow.keras as keras
import time

import Math_Utilities as math_utils
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
    frames = time_feats.Frames_fixed_length(FILE)               # create frames   
    waveform_RMS = math_utils.RMS_Energy(FILE.waveform)         # waveform RMS
      
    # Add Time-Domain Features
    FILE = FILE.add_features(time_feats.rise_decay_time(FILE))          # add rise & decay 
    FILE = FILE.add_features(waveform_RMS)                              # add RMS for full waveform
    FILE = FILE.add_features(time_feats.RMS_above(frames,waveform_RMS)) # add % above RMS

    # Add Frequency-Domain Features
    

    return FILE                    # return file with feature vector

def construct_targets (fileobjs,matrix=True):
    """
    Construct target array object 
    --------------------------------
    fileobjs (iter) : List of object instances w/ 'target' attribute
    matrix (bool) : If true, targets are one-hot-encoded into matrix,
        else a vector is returned
    --------------------------------
    Return target object and nmber of unique classes
    """
    y = np.array([x.target for x in fileobjs])  # use target attribute
    n_classes = len(np.unique(y))               # number of classes
    if matrix == True:                          # if one-hot-enc
        y = keras.utils.to_categorical(y,n_classes)
    return y,n_classes                          # return target & classes

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
        print("\tFile:",FILE.filename)      # Current file
        FILE = FILE.read_audio()            # read .WAV file       
        FILE = Assemble_Features(FILE)      # gather features
        X = np.append(X,FILE.__getfeatures__())     # add row to design matrix
        del(FILE)
    X = X.reshape(n_samples,-1)     # reshape
    return X                        # return design matrix    
        
