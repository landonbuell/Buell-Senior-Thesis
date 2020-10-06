"""
Landon Buell
PHYS 799
Instrument Classifier v0
13 June 2020
"""

            #### IMPORTS ####

import numpy as np
import tensorflow.keras as keras

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

    def AddChannel (self):
        """ Add Extra dimension to Design Matrix - used for Spectrogram """
        currentShape = self.X.shape     # current shape
        newShape = [i for i in currentShape] + [1]
        self.X = self.X.reshape(newShape)
        return self

    def ShapeBySample (self):
        """ Reshape design matrix by number of samples """
        self.X = np.array(self.X)
        sampleShape = self.X[0].shape       # shape of 1 sample
        newShape = [self.n_samples] + [i for i in sampleShape]
        self.X = self.X.reshape(newShape)
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

