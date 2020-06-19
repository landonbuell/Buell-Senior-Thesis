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

import Program_Utilities as prog_utils
import Math_Utilities as math_utils
import Timespace_Features as time_feats
import Freqspace_Features as freq_feats
import Plotting_Utilities as plot_utils

            #### FUNCTION DEFINITIONS ####

def Assemble_Features (FILE):
    """
    Create & Collect all classification features
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return (1 x N) array of features
    """   

    # Features Pre-processing
    frames = time_feats.Frames_fixed_length(FILE.waveform)      # create frames    
    waveform_RMS = math_utils.RMS_Energy(FILE.waveform)         # waveform RMS
    f,pts = freq_feats.Frequency_Axis(rate=FILE.rate)           # frequency axis
    f,t,Sxx = freq_feats.Spectrogram(frames,f,pts)              # build spectrogram
    ESDs = freq_feats.Energy_Spectral_Density(f,t,Sxx,
                bands=[(0,32),(32,64),(64,128),(128,256),
            (256,512),(512,1024),(2048,4096),(4096,6000)])      # Energy in bands
      
    # Create Feature vector object
    FEATURES = prog_utils.feature_vector(FILE.target)
    FEATURES = FEATURES.add_features(time_feats.Rise_Decay_Time(FILE.waveform))
    FEATURES = FEATURES.add_features(math_utils.RMS_Energy(FILE.waveform))
    FEATURES = FEATURES.add_features(ESDs)

    return FEATURES

def construct_targets (objs,matrix=True):
    """
    Construct target array object 
    --------------------------------
    objs (iter) : List of object instances w/ 'target' attribute
    matrix (bool) : If true, targets are one-hot-encoded into matrix,
        else a vector is returned
    --------------------------------
    Return target object and nmber of unique classes
    """
    y = np.array([x.target for x in objs])      # use target attribute
    n_classes = len(np.unique(y))               # number of classes
    if matrix == True:                          # if one-hot-enc  
        y = keras.utils.to_categorical(y,24)
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
        print("\t("+str(I)+"/"+str(n_samples)+")",FILE.filename)      # Current file
        FILE = FILE.read_audio()            # read .WAV file       
        FEATURES = Assemble_Features(FILE)          # gather features
        X = np.append(X,FEATURES.__getfeatures__()) # add row to design matrix
        del(FILE)
    X = X.reshape(n_samples,-1)     # reshape
    return X                        # return design matrix    
        
