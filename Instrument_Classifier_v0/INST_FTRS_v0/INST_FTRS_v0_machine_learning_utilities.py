"""
Landon Buell
Instrument Classifier v0
Machine Learning Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import INST_FTRS_v0_base_utilities as base_utils
import INST_FTRS_v0_feature_utilities as feat_utils
 

"""
INSTRUMENT FEATURES V0 - MACHINE LEARNING UTILITIES
        Functions related to machine learning structure & workflow
    - split_train_test
    - target_label_encoder
    - Design_Matric_Scaler
    - Design_Matrix
    - Create_MLP_Model
    - Confusion_Matrix
"""

            #### PREPROCESSING FUNCTIONS ####

def split_train_test (X,y,test=0.25,seed=None):
    """
    generate a series of indicies for training & testing data
        Adapted from (Geron, 49) (Note: numpy is Psuedo-Random)
    --------------------------------
    X (arr) : Design matrix, (n_samples x n_features)
    y (arr) : Target vector, (n_samples x 1)
    test (float) : ratio of testing data size on the bound (0,1)
    seed (int) : Random state for split
    --------------------------------
    return lists of training obj instances & testing obj instances 
    """
    return train_test_split(X,y,test_size=test,random_state=seed)


def target_label_encoder(target_vector,write=False):
    """
    Create encoding dictiory of strings to classes
    --------------------------------
    target_vector (arr) : array of target classes as strings
    write (bool/str) : If not False, str is path write out dict 
    --------------------------------
    Return encoding & decoding dictionary
    """
    enc_dict = {}                       # output dictionary
    class_counter = 0                   # class counter
    for instrument in np.unique(target_vector): # unique elements
        key,val = instrument,class_counter
        enc_dict.update({key:val})      # update the dictionary
        class_counter += 1              # incriment class counter
    dec_dict = {value:key for key,value in enc_dict.items()}

    if write != False:                   # if write out dictionary
        decode = pd.DataFrame(data=np.array(list(dec_dict.items())),
                              index=np.arange(0,class_counter,1),
                              columns=['Target','Instrument'])
        decode.to_csv(write+'/DECODE.csv')

    return enc_dict,dec_dict            # return the encoding/decoding dictionary

def Design_Matrix_Scaler (X):
    """
    Scale design matrix to have 0 mean, & unit variance
    --------------------------------
    X (arr) : (n_samples x n_features) standard Design matrix to scale
    --------------------------------
    Return scaled design matrix
    """
    scaler = StandardScaler()   # obj inst
    X = scaler.fit_transform(X) # fit the inst & tranf X
    return X

def Design_Matrix_Labeler (X,inc_cols=True):
    """
    Clean and label columns of design matrix
    --------------------------------
    X (arr) : Standard format design matrix (n_samples x n_features)
    inc_cols (bool) : Attatch hardcoded list of column names to frame if True
    --------------------------------
    Return design matrix as pandas DataFrame
    """
    cols = ['Rise Time','Decay Time','RMS Energy',
            '>10% RMS','>20% RMS','>30% RMS','>40% RMS','>50% RMS',
            '>60% RMS','>70% RMS','>80% RMS','>90% RMS',
            'Band 1 Energy','Band 2 Energy','Band 3 Energy','Band 4 Energy',
            'Band 5 Energy','Band 6 Energy','Band 7 Energy','Band 8 Energy',]
    if cols == True:
        return pd.DataFrame(data=X,columns=cols)
    else:
        return pd.DataFrame(data=X)


def Design_Matrix (wavfile_objects,wav_path,int_path):
    """
    Construct feature matrix "X" (n_samples x n_features) 
    --------------------------------
    wavfile_objects (list) : List of wavfile object instances to read
    wav_path (str) : Full directory path where training .wav files are stored
    int_path (str) : Full directory path where program is stored
    --------------------------------
    Return feature matrix "X" (n_samples x n_features) 
    """
    # Setup & intialize
    n_samples = len(wavfile_objects)        # number of samples

    # Build Feature Matrix
    X = np.array([])                        # feature matrix
    
    for WAVFILE in wavfile_objects:         # through each .wav file
        print('\t',WAVFILE.filename)
        os.chdir(wav_path)                  # change to directory
        WAVFILE = WAVFILE.read_raw_wav()    # read waveform as np array
        os.chdir(int_path)                  # change to home directory

        timeseries_features = feat_utils.timeseries(WAVFILE)     # collect time features
        freqseries_features = feat_utils.freqseries(WAVFILE)     # colelct freq features

        row = np.array([])          # feature vector for sample
        row = np.append(row,timeseries_features)    # add time features
        row = np.append(row,freqseries_features)    # add freq features
        X = np.append(X,row)        # add to feature matrix

    return X.reshape(n_samples,-1)     # rehape feature matrix

