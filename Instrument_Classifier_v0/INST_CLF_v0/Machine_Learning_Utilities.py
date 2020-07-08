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
import Timespace_Features as time_feats
import Freqspace_Features as freq_feats
import Plotting_Utilities as plot_utils
import Math_Utilities as math_utils
import Neural_Network_Models

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
    N = int(2**12)                                              # length of audio frame
    frames = time_feats.Frames_fixed_length(FILE.waveform,npts=N)  # create frames    
    waveform_RMS = math_utils.RMS_Energy(FILE.waveform)         # waveform RMS
    frames_RMS = math_utils.RMS_Energy(frames)                  # frames RMS
    f,pts = freq_feats.Frequency_Axis(rate=FILE.rate,npts=N)    # frequency axis
    f,t,Sxx = freq_feats.Spectrogram(frames,f,pts)              # build spectrogram
    ESDs = freq_feats.Energy_Spectral_Density(f,t,Sxx,
                bands=[(0,32),(32,64),(64,128),(128,256),
            (256,512),(512,1024),(2048,4096),(4096,6000)])      # Energy in bands
      
    # Feature vector for Perceptron model
    MLP_feats = prog_utils.Feature_Array(FILE.target)    # create instance
    MLP_feats = MLP_feats.add_features(time_feats.Rise_Decay_Time(FILE.waveform))
    MLP_feats = MLP_feats.add_features(waveform_RMS)      
    MLP_feats = MLP_feats.add_features(math_utils.Distribution_Features(frames_RMS))
    MLP_feats = MLP_feats.add_features(ESDs)

    # Feature vector for Spectrogram_Classifier
    Sxx_feats = prog_utils.Feature_Array(FILE.target)    # create instance
    Sxx_feats = Sxx_feats.set_features(Sxx)     
    
    # Feature vector for Phase-Space Classifier
    PSC_feat_set = []                   # set of feature objects
    phase_matrices = time_feats.Phase_Space(frames)
    PSC_feats = prog_utils.Feature_Array(FILE.target)    # create instance
    """
    Need to finish Phase-Space Features design process
        Maybe used 3D convolution? May be worth looking into...
    """ 
    # return feature objects
    return MLP_feats,Sxx_feats,PSC_feat_set

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
    MLP_matrix = prog_utils.Design_Matrix(ndim=2)   # design matrix for perceptron
    SXX_matrix = prog_utils.Design_Matrix(ndim=3)   # design matrix for spectrogram
    PSC_matrix = prog_utils.Design_Matrix(ndim=3)   # design matrix for phase-space

    for I,FILE in enumerate(FILE_OBJECTS):  # iterate through files
        print('\t\t\t('+str(I)+'/'+str(n_samples)+')',FILE.filename)
        FILE = FILE.read_audio()            # read .WAV file

        MLP_feats,Sxx_feats,PSC_feat_set = \
            Assemble_Features(FILE)         # asssmble features from file

        MLP_matrix.add_sample(MLP_feats)    # add features to MLP matrix
        SXX_matrix.add_sample(Sxx_feats)    # add features to Sxx matrix
        for PSC_feats in PSC_feats_set:     
            PSC_matrix.add_sample(PSC_feats)# add features to PSC matrix

    MLP_matrix = MLP_Matrix.shape_by_sample()
    SXX_matrix = SXX_matrix.pad_2D(new_shape=Neural_Network_Models.spectrogram_shape)
    PSC_matrix = PSC_matrix.pad_2D(newshape=Neural_Network_Models.phasespace_shape)

    return [MLP_matrix,SXX_matrix,PSC_matrix]   # return design matrix objs

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

    
      