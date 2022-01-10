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

import CommonStructures

        #### CLASS DEFINTIONS ####

class SelectKBest:
    """ Select K Features that score best by metric """
    
    def __init__(self):
        """ Constructor for SelectKBest """


class FeatureScaler:
    """ Scales each Feature of a Design Matrix to Have Zero Mean and Unit Variance """

    def __init__(self):
        """ Constructor for FeatureScaler Instance """
        self._sampleShape = []
        self._matrixType = "-1"
        self._isFitted = False
        self._means = np.array([])
        self._varis = np.array([])

    def __del__(self):
        """ Destructor for FeatureScaler Instance """
        self._runInfo = None

    # Getters and Setters

    def getNumFeatures(self):
        """ Return the Number of Features in the Design Matrix """
        return self._numFeatures

    def getMeans(self):
        """ Get the Variance of Each Feature """
        return self._means

    def getVariances(self):
        """ Return the Variance of Each Feature """

    # Public Interface

    def fitRunInfo(self,runInfoInstance,matrixType="A"):
        """ Fit the Scaler by Using a RunInfo Instance (Larger DataSets) """
        if (matrixType == "A"):
            # Type A Matrix
            self._sampleShape = np.array(runInfoInstance.getShapeSampleA())    
            self._matrixType = "A"
        elif( matrixType == "B"):
            # Type B Matrix
            self._sampleShape = np.array(runInfoInstance.getShapeSampleB())
            self._matrixType = "B"
        else:
            # Bad Matrix Type
            raise ValueError("matrixType must be A or B!")
        return self

    def fitDesignMatrix(self,designMatrix):
        """ Fit the Scaler by Using a DesignMatrix Instance (Smaller DataSets) """
        self._sampleShape = designMatrix.getSampleShape()
        self._means = np.mean(designMatrix.getFeatures(),axis=0,dtype=np.float32)
        self._varis = np.var(designMatrix.getFeatures(),axis=0,dtype=np.float32)
        self._isFitted = True
        return self

    def transform(self,designMatrix,verbose=True):
        """ Transform Design Matrix According to Scaling Parameters """
        if (self._isFitted == False):
            raise RuntimeError("Scaler not yet fit")
        dataRef = designMatrix.getFeatures()
        dataRef = (dataRef - self._means) / np.sqrt(self._varis)
        designMatrix.setFeatures(dataRef)

        # Print to Console?
        if (verbose == True):
            print(np.mean(designMatrix.getFeatures(),axis=0,dtype=np.float32))
            print(np.var(designMatrix.getFeatures(),axis=0,dtype=np.float32))
        return designMatrix


    def clear(self):
        """ Reset the State of the Instance """
        self._sampleShape = []
        self._matrixType = "-1"
        self._isFitted = False
        self._means = np.array([])
        self._varis = np.array([])
        return self

