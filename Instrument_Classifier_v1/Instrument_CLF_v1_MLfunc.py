"""
Landon Buell
Instrument Classifier v1
Machine Learning Functions
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

import Instrument_CLF_v1_func as func
import Instrument_CLF_v1_features as features

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

def prediction_function (CLF,X):
    """ 
    Make predictions about input data X
    --------------------------------
    CLF (class) : Trained classifier model instance to test on
    X (array) : feature matrix to make prediction on
    --------------------------------
    return final prediction of data X on CLF
    """
    confidence = CLF.decision_function(X)   # conficidence scores on X matrix
    # conf is (n_samps x n_classes)         
    scores = np.sum(confidence,axis=0,dtype=int)
    prediction = np.argmax(scores)          # prediction for X matrix 
    return prediction                       # return vals

            #### CLASSIFIER OBJECTS ####

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
    for name in names:                          # for each desired classifier
        CLF = SGDClassifier(random_state=seed,
                max_iter=1000,tol=1e-3)         # create classifier
        setattr(CLF,'name',str(name))           # attach name tp classifier
        pair = {str(name):CLF}                  # name calls to specific clf inst
        classifier_dictionary.update(pair)      # add to dictionary
    return classifier_dictionary                # return the dictionary

def LogReg_CLFs (names,seed=None):
    """
    Create dictionary of Logisitc Regression Classifier Objects
    --------------------------------
    name (list) : name to attach to each SGD object
    seed (int) : seed nunber to use for reproduceable results (None by default)
    --------------------------------
    returns SGD Object w/ name
    """  
    classifier_dictionary = {}                  # empty dictionary
    for name in names:                          # for each desired classifier 
        CLF = LogisticRegression(random_state=None,
                max_iter=100,tol=1e-4)
        setattr(CLF, 'name', name)              # attatch name to classifier
        pair = {str(name):CLF}                  # create dict pair
        classifier_dictionary.update(pair)      # add to dictionary
    return classifier_dictionary                # return dictionary
    
def train_classifiers (wavfiles,clf_dict,read_dir,home_dir,classes):
    """
    Train Classifiers on set of file obejct instances
    --------------------------------
    wavfiles (inst) : instance of .wav file to train on classifiers
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    read_dir (str) : local directory path where raw .wav files are
    home_dir (str) : local directory path where program is based
    classes (array) : array of class labels
    --------------------------------
    Returns classifier dictionary object
    """
    n_samples = len(wavfiles)               # number of samples
    X,y = np.array([]),np.array([])         # X matrix & target vector
    for wavfile in wavfiles:                # each training instance 
        sample_features = np.array([])      # row of features for instance
        os.chdir(read_dir)                  # change to wav directory
        wavfile.read_raw_wav()              # read waveform (add attrb)
        os.chdir(home_dir)                  # intial dir
        # FUNCTION TO PULL OUT N FEATURES FROM TIME SERIES SINGLE WAV FILE INSTANCE    
        x = features.timeseries_features(wavfile)
        # FUNCTION TO PULL OUT N FEATURES FROM FREQ SERIES SINGLE WAV FILE INSTANCE 
        x = None

        y = np.append(y,wavfile.class_num)  # add to target vector
        del(wavfile.data)                   # delete waveform
    X = X.reshape(n_samples,-1)             # reshape n_samps x n_features
    return clf_dict                         # return the classifier dictionary

def test_classifiers (wavfiles,clf_dict,read_dir,home_dir,classes):
    """
    test Classifiers on set of file obejct instances
    --------------------------------
    wavfiles (inst) : instance of .wav file to train on classifiers
    clf_dict (dict) : Dictionary of classifiers to be partially fit w. data
    read_dir (str) : local directory path where raw .wav files are
    home_dir (str) : local directory path where program is based
    classes (array) : array of class labels
    --------------------------------
    Returns array of actual values & predicted values
    """
    ytrue,ypred = np.array([]),np.array([])
    for wavfile in wavfiles:                # each testing instance
        #print("\t\t",wavfile.filename)      # print filename
        os.chdir(read_dir)                  # change to wav directory
        wavfile.read_raw_wav()              # read waveform (add attrb)
        os.chdir(home_dir)                  # intial dir
        actl,pred = features.test_wavfile(wavfile,clf_dict,classes)
        del(wavfile.data)                   # delete waveform
        ytrue = np.append(ytrue,actl)       # add actual class
        ypred = np.append(ypred,pred)       # add predicted value
    return ytrue,ypred

            #### METRICS ####

def confusion_matrix (title,ytrue,ypred,labs,save=False,show=False):
    """ 
    Produce sklearn confusion matrix for classifier predictions 
    Can plot to console or save to current local path
    --------------------------------
    title (str) : Title for confusion matrix plot
    ytrue (arr) : (1 x N) size array of actual instance labels
    ypred (arr) : (1 x N) size array of classifier predictions
    labs (arr) : (1 x M) array of class labels
    --------------------------------
    Return (n_classes x n_classes) confusion matrix
    """
    matrix = metrics.confusion_matrix(ytrue,ypred)
    if show == True or save == True:
        plt.title(str(title),size=40,weight='bold')
        plt.imshow(matrix,cmap=plt.cm.binary)
        plt.xticks(labs)
        plt.yticks(labs)
        plt.xlabel('Actual Class',size=20,weight='bold')
        plt.ylabel('Predicted Class',size=20,weight='bold')
        if save == True:
            plt.savefig(str(title)+'.png')
        if show == True:
            plt.show()
    return matrix