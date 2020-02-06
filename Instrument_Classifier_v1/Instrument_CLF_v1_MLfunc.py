"""
Landon Buell
Instrument Classifier v1
Machine Learning Functions
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import os

from sklearn.linear_model import SGDClassifier
import sklearn.metrics as metrics

import Instrument_CLF_v1_func as func

"""
INSTRUMENT CLASSIFIER V1 - MACHINE LEARNING ALGORITHM RELATED FUNCTIONS


"""

            #### FUNCTIONS DEFINTIONS ####

def SGD_CLFs (names,seed=None):
    """
    Create dictionary of SGD Classifier Object
    --------------------------------
    name (list) : name to attach to each SGD object
    seed (int) : seed nunber to use for reproduceable results (None by default)
    --------------------------------
    returns SGD Object w/ name
    """
    classifier_dictionary = {}                  # empty dictionary
    for I in range(len(names)):                 # for each desired classifier
        CLF = SGDClassifier(random_state=seed)  # create classifier
        setattr(CLF,'name',str(name))           # attach name tp classifier
        pair = {str(name):CLF}                  # name calls to specific clf inst
        classifier_dictionary.update(pair)      # add to dictionary
    return classifier_dictionary                # return the dictionary

def split_train_test (nsamps,ratio):
    """
    generate a series of indicies for training & testing data
        Adapted from (Geron, 49) (Note: numpy is Psuedo-Random)
    --------------------------------
    nsamps (int) : number of sample data points
    ratio (float) : ratio of train: test data (0,1)
    --------------------------------
    return train / test indicices
    """
    shuffled = np.random.permutation(nsamps)    # permute idxs
    test_size = int(nsamps*ratio)               # test dataset size
    train = shuffled[test_size:].tolist()       # set training idxs
    test = shuffled[:test_size]                 # set testing idxs
    return train,test                           # return data pts


