"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           Preprocessing.py
 
Author:         Landon Buell
Date:           January 2022
"""

        #### IMPORTS ####

import numpy as np
from numpy.lib.function_base import average

import CommonStructures

        #### CLASS DEFINTIONS ####

class PreprocessingTool:
    """ Abstract Parent Class For all Preprocessing Tools """

    def __init__(self,toolName):
        """ Constructor for PreprocessingTool Abstract Class """
        self._toolName = toolName
        self._timesFit = 0
        self._sampleShape = []
        
    def __del__(self):
        """ Destruction for PreprocessingTool Abstract Class """
        pass

    # Getters and Setters

    def getToolName(self):
        """ Return the name of this tool """
        return self._toolName

    def getTimesFit(self):
        """ Get the Number of times that the tool has been fit """
        return self._timesFit

    def getIsFit(self):
        """ Return T/F if the model has been fit at all """
        return (self._timesFit > 0)

    def getSampleShape(self):
        """ Get the Shape of Each Sample """
        return self._sampleShape

    # Public Interfacce

    def fit(self,designMatrix):
        """ Fit the Preprocessing Tool w/ a design matrix Object """
        return self

    def transform(self,designMatrix):
        """ Transform a design matrix with the fir params """
        return designMatrix

    def reset(self):
        """ Reset the state of the tool """
        self._timesFit = 0
        self._sampleShape = []
        return self

    # Magic Methods

    def __repr__(self):
        """ Return Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class SelectKBest:
    """ Select K Features that score best by metric """
    
    def __init__(self,scoringCallback):
        """ Constructor for SelectKBest """
        super().__init__("SelectKBest")
        self._scoringCallback = scoringCallback

    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        pass


class FeatureScaler:
    """ Scales each Feature of a Design Matrix to Have Zero Mean and Unit Variance """

    def __init__(self):
        """ Constructor for FeatureScaler Instance """
        super().__init__("FeatureScaler")
        self._historicalWeights
        self._means = np.array([],dtype=np.float32)
        self._varis = np.array([],dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        pass

    # Getters and Setters

    def getAverageOfMeans(self):
        """ Compute the Average of all batch means """
        return np.mean(self._means,axis=0,dtype=np.float32)

    def getAverageOfVariances(self):
        """ Compute the Average of all batch variances """
        return np.mean(self._varis,axis=0,dtype=np.float32)

    # Public Interface

    def fit(self,designMatrix):
        """ Fit the Preprocessing Tool w/ a design matrix Object """
        featureMeans = self.computeBatchMeans(designMatrix)
        featureVaris = self.computeBatchVaris(designMatrix)
        self.storeBatchMeans(featureMeans)
        self.storeBatchVaris(featureVaris)
        
        self._timesFit += 1
        return self

    def transform(self,designMatrix):
        """ Transform a design matrix with the fir params """
        if (self.getIsFit() == False):
            # Tool not yet Fit
            raise RuntimeError(repr(self) + " is not yet fit")
        return designMatrix

    # Private Interface

    def computeBatchMeans(self,X):
        """ Compute the Average of Each Feature in the given batch """
        return np.mean(X.getFeatures(),axis=0,dtype=np.float32)
        
    def computeBatchVaris(self,X):
        """ Compute the Variance of Each Feature in the given batch """
        return np.var(X.getFeatures(),axis=0,dtype=np.float32)
   
    def storeBatchMeans(self,x):
        """ Store the average of each feature in this batch """
        if (self._means.shape[0] == 0):
            self._sampleShape = list(x.shape)
        if (x.shape != self._sampleShape):
            raise ValueError("Shape Mismatch")
        newShape = [self._means.shape[0]] + self._sampleShape
        self._means = np.append(self._means,x)
        self._means.reshape(newShape)
        return self

    def storeBatchVaris(self,x):
        """ Store the variances of each feature in this batch """
        if (self._varis.shape[0] == 0):
            self._sampleShape = list(x.shape)
        if (x.shape != self._sampleShape):
            raise ValueError("Shape Mismatch")
        newShape = [self._varis.shape[0]] + self._sampleShape
        self._varis = np.append(self._varis,x)
        self._varis.reshape(newShape)
        return self


