"""
Landon Buell
Frequency Classifer v0
Functions
1 January 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
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

def read_csvfile (filename,disp=True):
    """
    Load data set from directory 
    --------------------------------
    filename (str) : string to identify column to use
    --------------------------------
    Returns matrix of CSV data
    """
    frame = pd.read_csv(filename,sep='\t',header=0)     # read CSV file
    frame.drop(frame.columns[0],axis=1,inplace=True)    # drope 1st column
    frame = frame.astype(dtype=float)                   # cast to floats
    if disp == True:                        # print output?
        print("\tDataFrame information:")
        print("\t\tSamples:",frame.shape[0])
        print("\t\tFeatures:",frame.shape[1])
    return frame                            # return the frame

def save_csvfile (filename,data,classes):
    """
    Load data set from directory 
    --------------------------------
    filename (str) : string to write file to
    data (arr) : N x N array of data to make into DataFrame
    classes (arr) : 1 x N array of class value indentifiers
    --------------------------------
    Returns matrix of CSV data
    """
    cols = ['Predicted: '+str(x) for x in classes]
    rows = ['Observed: '+str(x) for x in classes]
    frame = pd.DataFrame(data,index=rows,columns=cols,dype=float)
    frame.to_csv(filename,sep='\t',header=True,
                 index=Flase,mode='w')
    return frame

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
    xtrain,xtest = model.train_test_split(xdata,test_size=size,random_state=state)
    ytrain,ytest = model.train_test_split(ydata,test_size=size,random_state=state)
    data_dict = {'xtrain':xtrain,'xtest':xtest,
                 'ytrain':ytrain,'ytest':ytest}
    return data_dict                # return the dictionary of data sets

def stratified_split (xdata,ydata,size=0.1,state=0):
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
    ytrain (arr) : 1 x N array of training target classification values 
    state (int) : random state for classifier object (None by default)
    --------------------------------
    Return SGD Classifier
    """
    clf = SGDClassifier(random_state=state,max_iter=100)       
    setattr(clf,'name',name)                        # attach name of obj
    clf.fit(xtrain,ytrain)                          # train the classifier
    return clf

def confusion_matrix (clf,xdata,ydata,disp=True):
    """
    Compute confusion matrix for N-Classes Classifier object
    ----------------
    clf (classifier obj) : Classifier object to build confusion matrix for
    xdata (array/DataFrame) : x-training dataset
    ydata (array/DataFrame) : y-training target dataset
    disp (bool) : Visualize Confusion matrix (True by default)
    ----------------
    returns Confusion matrix
    """
    ypred = model.cross_val_predict(clf,xdata,ydata)    # cross-val prediction
    conf_mat = metrics.confusion_matrix(ydata,ypred)    # build conf matrix
    if disp == True:
        axes = np.arange(0,19)          # array for axes
        plt.matshow(conf_mat,cmap=plt.cm.gray)      # disp.matrix
        title = 'Confusion_Matrix3_'+str(clf.name)   # title for figure
        plt.title(title,size=12,weight='bold')      # set title
        plt.xticks(axes)            # set x ticks
        plt.yticks(axes)            # set y ticks
        plt.savefig(title+'.png')   # save to directory
        plt.show()       
    return conf_mat                 # return matrix

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
    metricsdict = {'Precision Score':precision,
                   'Recall Score':recall, 'F1 Score':f1}
    if disp == True:                                # print output to line?
        print("\tClassifier Metrics")               # header
        for keys,vals in metricdict.items():        # for each key value pair:
            print('\t\t'+str(keys),':',str(vals))   # print metric out
    return metricsdict