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

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
import sklearn.metrics as metrics

import tensorflow
from tensorflow import keras

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

def One_Hot_Encoder (y):
    """
    Convert target vector y into one-hot-encoded matrix
    --------------------------------
    y (arr) : target vector for design matrix
    --------------------------------
    return one-hot-encoded matrix Y, with number of unique classes
    """
    n_classes = np.unique(y).shape[0]   # unqiue classes
    Y = keras.utils.to_categorical(y,n_classes,dtype='uint')
    return Y,n_classes                  # return Y & num classes

def target_label_decoder(path,filename='DECODE.csv'):
    """
    Create encoding dictiory of strings to classes
    --------------------------------
    path (str) : Local path were decoding dictionary is stored
    filename (str) : Name of Local file path
    --------------------------------
    Return encoding & decoding dictionary
    """
    frame = pd.read_csv(path+'/'+filename,index_col=0).to_numpy()
    frame = frame.transpose()           # transp
    decode = {}                         # empty decode dict
    for key,val in zip(frame[0],frame[1]):
        decode.update({key:val})          # add pair
    return decode                       # return dictionary

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

            #### MODEL CONTRUCTION FUNCTIONS ####

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

def Create_Sequential_Model (name,n_features,n_classes,summary=False):
    """
    Create & Return constructed Keras Sequential Model
    --------------------------------
    name (str) : Name to attach to model at attribute
    --------------------------------
    Return Untrained model instance
    """
    # Initialize & Add input layer
    MODEL = keras.models.Sequential(name=str(name)) 
    MODEL.add(keras.layers.Input(shape=(n_features,),name='INPUT'))
    # Hidden layers
    MODEL.add(keras.layers.Dense(units=40,activation='relu'))
    MODEL.add(keras.layers.Dense(units=40,activation='relu'))
    # Output layers
    MODEL.add(keras.layers.Dense(units=n_classes,activation='softmax'))
    MODEL.compile(optimizer='sgd',loss='categorical_crossentropy')

    # Summary:
    if summary == True:
        print(MODEL.summary())

    return MODEL

            #### EVALUATION & METRICS ####

def Evaluate_Classifier (model,X_test,y_test,report=False):
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
    y_pred = np.argmax(model.predict(X_test),axis=-1)   # run prediction on model 
    # Compute & Attatch Confusion matrix
    confmat = metrics.confusion_matrix(y_test,y_pred)   # confmat
    setattr(model,'confusion',confmat)  # attatch
    # Compute & Attatch Precision scores
    precision = metrics.precision_score(y_test,y_pred,average=None)
    setattr(model,'precision',precision)
    # Compute & Attatch Recall scores
    recall = metrics.recall_score(y_test,y_pred,average=None)
    setattr(model,'recall',recall)

    # Classification Report
    if report == True:
        clf_rpt = metrics.classification_report(y_test,y_pred)
        print(clf_rpt)

    # Return model w/ attatched attrbs
    return model
