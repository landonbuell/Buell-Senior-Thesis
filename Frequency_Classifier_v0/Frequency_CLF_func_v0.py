"""
Landon Buell
Frequency Classifer v0
Functions
1 January 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os

from sklearn.linear_model import SGDClassifier
import sklearn.model_selection as model
import sklearn.metrics as metrics


            #### FUNCTIONS DEFINITIONS ####

def read_directory(dir,ext):
    """Read all files in given directory path"""
    output = np.array([])                   # array to hold filenames
    for roots,dirs,files in os.walk(dir):   # all objects in parent path
        for file in files:                  # files in list of files
            if file.endswith(ext):          # if match ext
                output = np.append(output,file)
    return output                           # return the list of files

def random_split (xdata,ydata,size=0.1,state=0):
    """
    Split dataset into training and testing data sets based on random seed
    --------------------------------
    xdata (arr) : N x M array of N samples, M features 
    ydata (arr) : 1 x N array of target classification values
    size (float) : float (0,1) indicating relative size of testing dataset
    state (int) : random state seed for random splitting of data
    --------------------------------
    Returns dictionary of train vs. test & x vs y data sets
    """ 
    xtrain,xtest = model.train_test_split(xdata,testsize=size,random_state=state)
    ytrain,ytest = model.train_test_split(xdata,testsize=size,random_state=state)
    data_dict = {'xtrain':xtrain,'xtest':xtest,
                 'ytrain':ytrain,'ytest':ytest}
    return data_dict                # return the dictionary of data sets

def startified_split (xdata,ydata,size=0.1,state=0):
    """
    Split dataset into training and testing data sets based on Stratification
    --------------------------------
    xdata (arr) : N x M array of N samples, M features 
    ydata (arr) : 1 x N array of target classification values
    size (float) : float (0,1) indicating relative size of testing dataset
    state (int) : random state seed for random splitting of data
    --------------------------------
    Returns dictionary of train vs. test & x vs y data sets
    """ 
    split = model.StratifiedShuffleSplit(n_splits=1,
                    test_size=size,random_state=state)
    train,test = split.split(xdata,ydata)       # train & test idxs
    data_dict = {'xtrain':xdata[train],'xtest':xdata[test],
                 'ytrain':ydata[train],'ytest':xdata[test]}
    return data_dict                            # return dict of data sets

def SGD_Classifier (name,xtrain,ytrain,state=None):
    """
    Create & train sklearn SGD object
    --------------------------------
    name (str) : Name to attatch to classifer for human identification
    xtrain (arr) : N x M array of training data - N samples, M features 
    ydata (arr) : 1 x N array of training target classification values 
    state (int) : random state for classifier object (None by default)
    --------------------------------
    Return SGD Classifier
    """
    clf = SGDClassifier(random_state=state)         # create classifer
    setattr(clf,'name',name)                        # attach name of obj
    clf.fit(xtrain,ytrain)                          # train the classifier
    return clf

def general_metrics (clf,xdata,ydata,disp=True):
    """
    Determine Precision, Recall & F1 Score for Classifier
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    disp (bool) : Display outputs to command line (True by default)
    ----------------
    returns Precision, Recall & F1 Score for Classifier
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    
    precision = metrics.precision_score(ydata,ypred)    # compute precision score
    recall = metrics.recall_score(ydata,ypred)          # compute recall score
    f1 = metrics.f1_score(ydata,ypred)                  # compute f1 score

    if disp == True:                        # print output to line?
        print('Precision Score:',precision)
        print('Recall Score:',recall)
        print("F1 Score:",f1)

    return precision,recall,f1                  # return values