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
from sklearn.feature_selection import SelectKBest
import sklearn.metrics as metrics

import INST_CLF_v0_base_utilities as base_utils
 
"""
INSTRUMENT FEATURES V0 - MACHINE LEARNING UTILITIES
        Functions related to machine learning structure & workflow
    - split_train_test
    - Design_Matric_Scaler
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

def K_Best_Features (X,y,K):
    """
    Run K-best features selection algorithm
    --------------------------------

    --------------------------------
    Return subset of X containing K best features
    """
    best = SelectKBest(k=K)
    best = best.fit(X,y)
    X_new = best.transform(X)
    print("Kept:",best.get_support())
    return X_new

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
            '>10% RMS','>25% RMS','>50% RMS','>75% RMS',
            'Band 1 Energy','Band 2 Energy','Band 3 Energy','Band 4 Energy',
            'Band 5 Energy','Band 6 Energy','Band 7 Energy','Band 8 Energy',]
    if inc_cols == True:
        return pd.DataFrame(data=X,columns=cols)
    else:
        return pd.DataFrame(data=X)

def Create_MLP_Model (name,layers,seed=None):
    """
    Create sklearn Multilayer Percepton model instance
    --------------------------------
    name (str) : name to attach to instance
    layers (tuple) : Hidden layer sizes
    seed (int) : Random state for classifier model (default=None)
    --------------------------------
    Return initialized MLP model instance
    """
    model = MLPClassifier(hidden_layer_sizes=layers,activation='relu',
                        solver='sgd',batch_size=100,max_iter=500,
                        tol=1e-4,random_state=seed)
    setattr(model,'name',name)      # attach name attribute
    return model                    # return initialized model

            #### EVALUATION & METRICS ####

def Evaluate_Classifier (model,X_test,y_test):
    """
    Evaluate the performance of a classifier w/ confusion matrix,
        precision score & recall score
    --------------------------------
    model (inst) : Instance of trained MLP classifer model
    X_test (arr) : Subset of design matrix for testing model
    y_test (arr) : Corresponding labels for design matrix
    --------------------------------
    Return model instances w/ metric values attached as attrbs
    """
    y_pred = model.predict(X_test)      # run prediction on model
    # Compute & Attatch Confusion matrix
    confmat = metrics.confusion_matrix(y_test,y_pred)   # confmat
    setattr(model,'confusion',confmat)  # attatch
    # Compute & Attatch Precision scores
    precision = metrics.precision_score(y_test,y_pred,average=None)
    setattr(model,'precision',precision)
    # Compute & Attatch Recall scores
    recall = metrics.recall_score(y_test,y_pred,average=None)
    setattr(model,'recall',recall)
    # Return model w/ attatched attrbs
    return model
