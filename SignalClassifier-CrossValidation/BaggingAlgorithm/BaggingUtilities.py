"""
Landon Buell
Kevin Short
PHYS 799
18 October 2020
"""

        #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import tensorflow.keras as keras

import PlottingUtilities as plotutils

        #### CLASS DEFINITIONS ####

class ProgramSetup :
    """ Intialize Program, Organize Paths, Etc. """

    def __init__(self,modelName,dataPath,modelPath):
        """ Initialize ProgramSetup Instance """
        self.modelName = modelName
        self.dataPath = dataPath
        self.modelPath = modelPath

    def __Call__(self):
        """ Run Instance of Program Setup """
        self.modelPathList = self.GetModelPaths()
        self.trainingFiles = self.GetKeyWordFiles("@TRAINING-HISTORY@")
        self.predictionFiles = self.GetKeyWordFiles("@PREDICTIONS@")
        return self

    @property
    def GetPathLists (self):
        """ Return Lists Needed For Bagging Algorithm """
        return (self.modelPathList,self.trainingFiles,self.predictionFiles)

    def GetKeyWordFiles (self,keyWord):
        """ Collect All .csv files with 'keyWord' in it """
        files = []
        for item in os.listdir(self.dataPath):
            path = os.path.join(self.dataPath,item)           
            if (self.modelName in item) and (keyWord in item): 
                # this is a prediction file of this model
                files.append(path)      # add file to list
        return files

    def GetModelPaths (self):
        """ Collect Names of All Models in Folder """
        pathList = []
        for item in os.listdir(self.modelPath): # in the path
            path = os.path.join(self.modelPath,item)
            if os.path.isdir(path) and (self.modelName in item):
                # This is a folder that contains thedata for a tensorflow model          
                pathList.append(path)
        return pathList

class BaggingAlgorithm :
    """
    Execute Baggining Algorithm
    """

    def __init__(self,models,train,pred):
        """ Initialize BaggingAlgorithmInstance """
        self.modelPathList = models
        self.trainingFiles = train
        self.predictionFiles = pred
        self.nFiles = len(self.modelPathList)

    def LoadModel(self,modelPath):
        """ Load a save TensorFlow model """
        model = keras.models.load_model(modelPath)
        for i in range (len(model.layers)):
            layerWeights = model.layers[i].weights[0].numpy()
            layerBiases = model.layers[i].weights[1].numpy()

        return self

    def __Call__(self):
        """ Run Bagging Algorithm """
        
        for i in range(self.nFiles):
            modelPath = self.modelPathList[i]
            self.LoadModel(modelPath)

        return self

class Mathematics:
    """ Static methods to provide mathematical functions """

    @staticmethod
    def WeightLossScore (lossData,weights=None):
        """ Compute Weighted Loss Score of Loss Data History """
        _epochs = len(lossData)
        if weights is None:                     # no weight provided
            weights = np.arange(_epochs)  # linear weights
        weightedScore = np.dot(lossData,weights)/lossData.sum()
        return weightedScore

    @staticmethod
    def SoftmaxFunction(data):
        """ Apply Softmax function to array 'data' """
        sumExp = np.exp(data).sum()
        return np.exp(data)/sumExp



