"""
Landon Buell
Instrument Classifier v1
Classifier - Machine Learning Utility Functions
1 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

import INST_CLF_v1_base_utilities as base_utils

"""
INSTRUMENT CLASSIFIER v1 - MACHINE LEARNING UTILITIES
        Functions related to machine learning structure & workflow
    - split_train_test
    - target_label_encoder
    - Design_Matric_Scaler
    - Design_Matrix
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

def Scale_Design_Matrix (X):
    """
    Scale design Matrix so that Cols have mean=0, var=1
    --------------------------------
    X (arr/frame) : Standard design matrix (n_samples x n_features)
    --------------------------------
    Return scaled design matrix X
    """
    scaler = StandardScaler()   # create inst
    X = scaler.fit_transform(X) # fit the inst & tranf X
    return X                    # Return X

def one_hot_encoder (y):
    """
    One-Hot-Encode target vector y
    --------------------------------
    y (arr) : Standard target vector (n_samples x 1)
    --------------------------------
    Return encoded target Matrix Y (n_samples,n_classes)
    """
    y = y.ravel()                       # flatten
    n_classes = np.unique(y).shape[0]   # number of classes
    Y = keras.utils.to_categorical(y,n_classes)
    return Y,n_classes                  # return new matrix

            #### EVALUATION FUNCTIONS ####

def Evaluate_Model(model,X_test,y_test):
    """
    Evalaute Tensorflow Neural Network Model
    --------------------------------
    model (inst) : Trained NN Classifier Model instance
    X_test (arr) : Design Matrix of testing samples (n_samples x n_features)
    y_test (arr) : Target vector of testing samples (n_samples x 1)
    --------------------------------
    Return None
    """
    y_pred = np.argmax(model.predict(X_test),axis=-1)   # prediction
    # CLF Report
    report = metrics.classification_report(y_test,y_pred)
    print(report)
    # Conf-Mat
    confmat = metrics.confusion_matrix(y_test,y_pred)
    setattr(model,'confusion',confmat)
    base_utils.Plot_Confusion(model)
    return model