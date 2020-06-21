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
    Return: v_sample 3D array of features for phase-CNN
            w_sample 2D array of features for spect-CNN
            x_sample 1D array of features for class-MLP
    """   

    # Features Pre-processing
    frames = time_feats.Frames_fixed_length(FILE.waveform)      # create frames    
    waveform_RMS = math_utils.RMS_Energy(FILE.waveform)         # waveform RMS
    frames_RMS = math_utils.RMS_Energy(frames)                  # frames RMS
    f,pts = freq_feats.Frequency_Axis(rate=FILE.rate)           # frequency axis
    f,t,Sxx = freq_feats.Spectrogram(frames,f,pts)              # build spectrogram
    ESDs = freq_feats.Energy_Spectral_Density(f,t,Sxx,
                bands=[(0,32),(32,64),(64,128),(128,256),
            (256,512),(512,1024),(2048,4096),(4096,6000)])      # Energy in bands
      
    # Feature vector x for MLP model
    x_sample = prog_utils.Feature_Array(FILE.target)    # create instance
    x_sample = x_sample.add_features(time_feats.Rise_Decay_Time(FILE.waveform))
    x_sample = x_sample.add_features(waveform_RMS)      
    x_sample = x_sample.add_features(math_utils.Distribution_features(frames_RMS))
    x_sample = x_sample.add_features(ESDs)

    # Create Spectrogram feature object
    w_sample = prog_utils.Feature_Array(FILE.target)    # create instance
    w_sample = w_sample.set_features(Sxx.toarray().transpose())               
    
    # Create Phase-Space Feature object
    v_sample = prog_utils.Feature_Array(FILE.target)
    d1_frames = time_feats.Phase_Space(frames)
    d2_frames = time_feats.Phase_Space(d1_frames)
    v_features = np.array([frames,d1_frames,d2_frames])
    v_features = np.moveaxis(v_features,0,-1)
    v_sample.set_features(v_features)
    
    # return feature objects
    return v_sample,w_sample,x_sample

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
    n_classes = 25
    if matrix == True:                          # if one-hot-enc  
        y = keras.utils.to_categorical(y,n_classes)
    return y,n_classes                          # return target & classes

def Design_Matrices (FILE_OBJECTS):
    """
    Construct Standard Machine-Learning Design Matrix
        (n_samples x n_features)
    --------------------------------
    FILE_OBJECTS (iter) : Iterable of all file objects to use
    --------------------------------
    Return design matrix, X
    """
    n_samples = len(FILE_OBJECTS)           # number of file samples
    V = prog_utils.Design_Matrix(ndim=4)    # design matrix for Phase-space
    W = prog_utils.Design_Matrix(ndim=3)    # design matrix for spectrograms
    X = prog_utils.Design_Matrix(ndim=2)    # design matrix for perceptron

    for I,FILE in enumerate(FILE_OBJECTS):  # iterate through files
        print('\t('+str(I)+'/'+str(n_samples)+')',FILE.filename)
        FILE = FILE.read_audio()            # read .WAV file

        v,w,x = Assemble_Features(FILE)     # collect feature objects
        del(FILE)                           # delete file instance

        V = V.add_sample(v)   # add sample to phase design-matrix
        W = W.add_sample(w)   # add sample to spectrogram design-matrix
        X = X.add_sample(x)   # add sample to perceptron design-matrix

    return V,W,X                # return design matricies
    
        
