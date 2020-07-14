"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import time

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils
import Math_Utilities as math_utils
import Neural_Network_Utilities

            #### Dat Structrue 

class Design_Matrix:
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

    def add_sample (self,x):
        """ Add features 'x' to design matrix, preserve shape """
        self.X.append(x.__getfeatures__())      # add sample to design matrix
        self.shapes.append(x.__getshape__())    # store shape       
        self.n_samples += 1                     # current number of samples
        try:
            self.targets.append(x.target)       # add the target
        except:                                 # otherwise
            self.targets.append(None)           # add None-type
        return self
 
    def pad_2D (self,new_shape,offsets=(0,0)):
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

    def shape_by_sample (self,shape=None):
        """ Reshape design matrix by number of samples """
        if shape:
            self.X = self.X.reshape(shape)
        else:
            self.X = np.array(self.X).reshape(self.n_samples,-1)
        return self

    def scale_X (self,scaler):
        """ Apply standard preprocessing scaling to self.X """
        assert type(self.X) == np.ndarray
        return self

    def __get_Y__(self:
        """ treturn target matrix as One-hot-enc matrix """
        self.Y = keras.utils.to_categorical(self.targets,self.n_classes)
        return self.Y
           
    def __get_X__(self):
        """ return design matrix as rect. np array """
        return self.X

class Feature_Array:
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

    def add_features (self,x,axis=None):
        """ Add object x to feature vector attribute"""
        self.features = np.append(self.features,x,axis=axis)
        return self             # return self

    def set_features (self,x):
        """ Clear feature array, reset to object 'x' - preserve shape """
        self.features = x
        return self

    def reshape_arr (self,new_shape=(1,-1)):
        """ Reshape feature array to 'new_shape' """
        self.features = self.features.reshape(new_shape)
        return self


    def __getshape__(self):
        """ Return shape of feature attrb as tuple """
        return self.features.shape

    def __getfeatures__ (self):
        """ Assemble all features into single vector """
        return self.features    # return feature vector

    def __delfeatures__ (self):
        """ Delete all features (Save RAM) """
        self.features = np.array([])    # arr to hold features
        return self             # return new self

