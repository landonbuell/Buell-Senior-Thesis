"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras
import time

"""
MachineLearningUtilities.py - 'Machine Learning Utilities'
    Contains Definitions that are relatedto Machine Learning Data Structures & Workflow
    
"""

            #### Data Structrue ####

class DesignMatrix:
    """
    Construct design-matrix-like object
    --------------------------------
    target (int) : Integer target value
    ndim (int) : Number of dimensions in this array
    n_classes (int) : Number of unique classes 
    --------------------------------
    Return instantiated feature_array instance
    """

    def __init__(self,ndim,n_classes):
        """ Initialize Object Instance """
        self.X = []         # empty data structure
        self.shapes = []    # store explicit shapes of each samples
        self.targets = []   # target for each sample
        self.n_samples = 0  # no samples in design matrix
        self.ndim = ndim    # number of dimensions in array
        self.n_classes = n_classes

    def AddSample (self,x):
        """ Add features 'x' to design matrix, preserve shape """
        self.X.append(x.GetFeatures())      # add sample to design matrix
        self.shapes.append(x.GetShape())    # store shape       
        self.n_samples += 1                 # current number of samples
        try:                                # attempt
            self.targets.append(x.target)   # add the target
        except:                             # otherwise
            self.targets.append(None)       # add None-type
        return self
 
    def Pad2D (self,new_shape,offsets=(0,0)):
        """ Zero-Pad 2D samples to meet shape """
        new_X = np.zeros(shape=(self.n_samples,new_shape[0],new_shape[1]))   # create new design matrix
        new_shape = (new_shape[0],new_shape[1])
        for i in range(self.n_samples):     # iterate by sample
            dx,dy = offsets[0],offsets[1]   # align upper left     
            try: 
                new_X[i][dx:dx+self.X[i].shape[0],dy:dy+self.X[i].shape[1]] += self.X[i] 
            except:
                slice = self.X[i][:new_shape[0],:new_shape[1]]
                shape_diff = np.array(new_shape) - slice.shape      # needed padding
                slice = np.pad(slice,[[0,shape_diff[0]],[0,shape_diff[1]]])
                new_X[i] += slice
            self.shapes[i] = new_shape      # reset shape
        self.X = new_X              # overwrite
        self.X = self.X.reshape(self.n_samples,new_shape[0],new_shape[1],1)
        return self                         # return new instance

    def _Pad2D (self,newShape):
        """ Zero-Pad or crop 2D samples to meet shape """
        raise NotImplementedError()

    def ShapeBySample (self,shape=None):
        """ Reshape design matrix by number of samples """
        if shape:
            self.X = self.X.reshape(shape)
        else:
            self.X = np.array(self.X).reshape(self.n_samples,-1)
        return self

    def SetMatrixData(self,X_new):
        """ Set 'self.X' to given input X_new """
        self.X = X_new
        return self

    def __Get_Y__(self,onehot=True):
        """ treturn target matrix as One-hot-enc matrix """
        if onehot == True:
            self.Y = keras.utils.to_categorical(self.targets,self.n_classes)
        return self.Y
           
    def __Get_X__(self):
        """ return design matrix as rect. np array """
        return self.X

class FeatureArray:
    """
    Create Feature vector object
    --------------------------------
    target (int) : Integer target value for this sample
    --------------------------------
    Return instantiated feature_array instance
    """

    def __init__(self,target):
        """ Initialize Object Instance """
        self.target = target            # set target
        self.features = np.array([])    # arr to hold features

    def AddFeatures (self,x,axis=None):
        """ Add object x to feature vector attribute"""
        self.features = np.append(self.features,x,axis=axis)
        return self             # return self

    def SetFeatures (self,x):
        """ Clear feature array, reset to object 'x' - preserve shape """
        self.features = x
        return self

    def ReshapeArray (self,new_shape=(1,-1)):
        """ Reshape feature array to 'new_shape' """
        self.features = self.features.reshape(new_shape)
        return self

    def GetShape (self):
        """ Return shape of feature attrb as tuple """
        return self.features.shape

    def GetFeatures (self):
        """ Assemble all features into single vector """
        return self.features    # return feature vector

    def DelFeatures (self):
        """ Delete all features (Save RAM) """
        self.features = np.array([])    # arr to hold features
        return self             # return new self

class ModelAnalysis:
    """
    Analyze Performance of Neural Metwork models
    --------------------------------
    model_names (iter) : List-like of strings calling Network models by name  
    datapath (str) : Local path where data is contained
    n_classes (int) : number of discrete classes for models
    --------------------------------
    Return Instantiated Model Analysis Object
    """

    def __init__(self,modelName,datapath,n_classes):
        """ Instantiate Class Object """

        raise NotImplementedError()

        self.modelName = modelName      # name of NN models
        self.datapath = datapath        # path to find file
        self.n_classes = n_classes      # classes in model

        self.metrics = [keras.metrics.SparseCategoricalCrossentropy(),
                        keras.metrics.Precision(),
                        keras.metrics.Recall()]

        
        self.infile = self.datapath.split('\\')[-1] # last item in split
        outfile = self.infile.split('@')[-1]        # time-stamp + ext
        self.outfile_name = 'ANALYSIS@'+outfile

    def ReadData (self):
        """ Load in Data from external source """
        data = pd.read_csv(self.infile,header=0,index_col=0)
        return data

    def ModelMetrics (self,model):
        """ Compute all metrics for single model """
        y_pred = self.data[model]       # these are the predictions

    def __call__(self):
        """ Compute all metrics Values for all Models """
        self.data = self.read_data()        # get data from CSV
        self.truth = self.data['Label']     # get label column
        for model in self.model_names:      # each model
            self.model_metrics(model)       # compute metrics
