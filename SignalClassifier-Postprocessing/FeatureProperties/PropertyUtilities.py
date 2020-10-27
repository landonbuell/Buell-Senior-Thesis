"""
Landon Buell
Kevin Short
PHYS 799
26 october 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

            #### CLASS OBJECT DEFINITIONS ####

class FeatureProcessor:
    """
    Collect and process features by class 
    """

    def __init__(self,X,y,n_classes,n_features):
        """ Initialize Feature Processor Instance """
        self.X =  StandardScaler().fit_transform(X)
        self.y = y 
        self.nClasses = n_classes
        self.nFeatures = n_features
        outputFrame = np.empty(shape=(self.nClasses,self.nFeatures))

    def MakeBoxPlotData(self,data):
        """ Compute Data Need for BoxPlot """
        _min,_max = np.min(data),np.max(data)
        _Qs = np.quantile(data,[0.25,0.5,0.75])
        return np.array([_min,Qs,_max]).ravel()

    def __Call__(self):
        """ Execute Feature Processor Instance """

        # Make BoxPlot Data for the Full matrix
        for feature in range(self.nFeatures):   # Each feature:
            featureData = self.X[:,i]           # get col of matrix


        for i in range(self.nClasses):      # each class
            classRows = np.where(y == i)    # get rows
            classData = X[classRows]        # get features

class QuantileData :
    """

    """

    def __init__(self,data,classInt,classStr):
        """ Initialize QuantileData Instance """
        self.data = data
        self.classInt = classInt
        self.classStr = classStr
