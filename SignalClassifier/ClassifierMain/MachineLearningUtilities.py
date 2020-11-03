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

    def AddAxis (self):
        """ Add Extra dimension to Design Matrix - used for Spectrogram """
        currentShape = self.features.shape     # current shape
        newShape = [i for i in currentShape] + [1]
        self.features = self.features.reshape(newShape)
        return self

    @property
    def GetShape (self):
        """ Return shape of feature attrb as tuple """
        return self.features.shape

    @property
    def GetFeatures (self):
        """ Assemble all features into single vector """
        return self.features    # return feature vector

    def DelFeatures (self):
        """ Delete all features (Save RAM) """
        self.features = np.array([])    # arr to hold features
        return self             # return new self

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

    def __init__(self,shape,n_classes):
        """ Initialize Object Instance """
        self.shape = shape
        self.data = np.zeros(shape=shape,dtype=np.float64)     
        self.sampleShape = shape[1:]
        self.shapes = []    # store explicit shapes of each samples
        self.targets = []   # target for each sample
        self.n_samples = 0  # no samples in design matrix
        self.n_classes = n_classes

    def AddSample (self,featureObj,index):
        """ Add features 'x' to design matrix, preserve shape """
        self.data[index] = featureObj.GetFeatures
        self.shapes.append(featureObj.GetShape)     # store shape       
        self.n_samples += 1                         # current number of samples
        try:                                        # attempt
            self.targets.append(featureObj.target)  # add the target
        except:                                     # otherwise
            self.targets.append(None)               # add None-type
        return self

    def AddChannel (self):
        """ Add Extra dimension to Design Matrix - used for Spectrogram """
        currentShape = self.X.shape     # current shape
        newShape = [i for i in currentShape] + [1]
        self.data = self.data.reshape(newShape)
        return self

    def ShapeBySample (self):
        """ Reshape design matrix by number of samples """
        self.X = np.array(self.X)
        self.X.reshape(self.n_samples,-1)
        return self

    def SetMatrixData(self,X_new):
        """ Set 'self.X' to given input X_new """
        self.data = X_new
        return self

    def __GetY__(self,onehot=True):
        """ Return target matrix as One-hot-enc matrix """
        if onehot == True:
            self.Y = keras.utils.to_categorical(self.targets,self.n_classes)
        return self.Y

    def __GetX__(self):
        """ return design matrix as rect. np array """
        return self.data



