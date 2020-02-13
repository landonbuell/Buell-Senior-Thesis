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
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

import Instrument_CLF_v1_func as func

"""
INSTRUMENT CLASSIFIER V1 - MACHINE LEARNING ALGORITHM RELATED FUNCTIONS
    - Classifier Objects
    - splitting data sets
    - performance metrics
"""

            #### FUNCTIONS DEFINTIONS ####

def label_encoder (wavobjs):
    """
    Convert instrument name strings into 0-idx class number
        Uses sklearn.preprocessing.LabelEncoder
    --------------------------------
    wavobjs (list) : List of all instances of .wav file objects
    --------------------------------
    Returns list of all instances of .wav file
    """
    instruments = np.array([])          # array for inst names
    for wav in wavobjs:                 # for each instance
        instruments = np.append(instruments,wav.instrument)
    enc = LabelEncoder()                # instance of encoder
    class_labels = enc.fit_transform(instruments)   
    for I in range (len(class_labels)): # for each new label
        # attach each class num to each instance
        setattr(wavobjs[I],'class_num',class_labels[I])
    return class_labels                 # return the list for good measure

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
        CLF = SGDClassifier(random_state=seed,
                max_iter=1000,tol=1e-3)         # create classifier
        setattr(CLF,'name',str(names[I]))       # attach name tp classifier
        pair = {str(names[I]):CLF}              # name calls to specific clf inst
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

def Train_CLF (classifier,X,y):
    """
    Train a classier object with features and labels
    --------------------------------
    classifier (obj) : Classifier instance to train
    X (array) : Matrix of training data, (N x M)
    y (array) : vector of training labels (N x 1)
    --------------------------------
    Returns trained Classifier object
    """
    pass

            #### METRICS ####

def confusion_matrix (CLF,ytest,ypred,labs,show=False):
    """ Produce sklearn confusion matrix for classifier predictions """
    matrix = metrics.confusion_matrix(ytest,ypred)
    if show == True:
        plt.title(str(CLF.name),size=40,weight='bold')
        plt.imshow(matrix,cmap=plt.cm.binary)
        plt.xticks(labs)
        plt.yticks(labs)
        plt.xlabel('Actual Class',size=20,weight='bold')
        plt.ylabel('Predicted Class',size=20,weight='bold')
        plt.show()
    return matrix