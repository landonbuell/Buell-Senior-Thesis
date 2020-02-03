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

def split_train_test (nsamps,ratio):
    """
    generate a series of indicies for training & testing data
    Adapted from Geron, 49
    --------------------------------
    nsamps (int) : number of sample data points
    ratio (float) : ratio of train: test data (0,1)
    --------------------------------
    return train / test indicices
    """
    shuffled = np.random.permutation(nsamps)    # permute idxs
    test_size = int(nsamps*ratio)               # test dataset size
    train = shuffled[test_size:].tolist()                # set training idxs
    test = shuffled[:test_size]                 # set testing idxs
    return train,test                           # return data pts


def SGD_CLF (name,seed):
    """
    Create SGD Classifier Object
    --------------------------------
    name (str) : name to attach to SGD object
    seed (int) : seed nunber to use for reproduceable results
    --------------------------------
    returns SGD Object w/ name
    """
    CLF = SGDClassifier(random_state=seed)  # create classifier
    setattr(Clf,'name',name)                # attach to name
    return CLF                              # return classifier obj