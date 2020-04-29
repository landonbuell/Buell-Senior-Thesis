"""
Landon Buell
Instrument Classifier v0
Machine Learning Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import os

from sklearn.neural_network import MLPClassifier

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_feature_utilities as feat_utils
 

"""
INSTRUMENT CLASSIFIER V0 - MACHINE LEARNING UTILITIES


"""

            #### FUNCTION DEFINTIONS ####

def split_train_test (data,tt_ratio=0.6):
    """
    generate a series of indicies for training & testing data
        Adapted from (Geron, 49) (Note: numpy is Psuedo-Random)
    --------------------------------
    data (iterable) : list or collection of data to split
    ratio (float) : ratio of train: test data on the bound (0,1)
    --------------------------------
    return lists of training obj instances & testing obj instances 
    """
    n_samples = len(data)                       # number of samples in data
    shuffled = np.random.permutation(n_samples) # permute idxs
    train_size = int(n_samples*tt_ratio)        # test dataset size
    # create lists of indexes for training/testing
    train_pts = shuffled[:train_size].tolist()      
    test_pts = shuffled[train_size:].tolist()
    # breaking into list of training & testing data
    training_data = [data[x] for x in train_pts]
    testing_data = [data[x] for x in test_pts]
    return training_data,testing_data           # return the two lists

def target_label_encoder(target_vector):
    """
    Create encoding dictiory of strings to classes
    --------------------------------
    target_vector (arr) : array of target classes as strings
    --------------------------------
    Return encoding dictionary
    """
    enc_dict = {}                       # output dictionary
    class_counter = 0                   # class counter
    for instrument in np.unique(target_vector): # unique elements
        key,val = instrument,class_counter
        enc_dict.update({key:val})      # update the dictionary
        class_counter += 1              # incriment class counter
    dec_dict = {value:key for key,value in enc_dict.items()}
    return enc_dict,dec_dict            # return the encoding/decoding dictionary

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


            #### CREATE MODEL INSTANCES ####

def Create_MLP_Model (name,layers,seed=None):
    """
    Create sklearn Multilayer Percepton model instance
    --------------------------------
    name (str) : name to attach to instance
    layers (tuple) : Hidden layer sizes
    seed (int) : Random state for classifier model (default=None)
    --------------------------------

    """
    model = MLPClassifier(hidden_layer_sizes=layers,activation='relu',
                        solver='sgd',batchsize=10,max_iter=400,
                        tol=1e-4,random_state=seed)
    setattr(model,'name',name)      # attach name attribute
    return model                    # return initialized model