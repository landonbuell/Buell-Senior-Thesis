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

    def __init__(self,dataPath,modelPath):
        """ Initialize ProgramSetup Instance """
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.predictionFiles = self.GetKeyWordFiles("@PREDICTIONS@")
        self.trainingFiles = self.GetKeyWordFiles("@TRAINING-HISTORY@")

    def GetKeyWordFiles (self,keyWord):
        """ Collect All .csv files with 'keyWord' in it """
        files = []
        for file in os.listdir(self.dataPath):  # in the path
            if file.endswith(".csv"):           # is csv file:
                if keyWord in file:             # keyword in name
                    files.append(file)          # add file to list
        return files

    def GetModelNames (self):
        """ Collect Names of All Models in Folder """
        modelNames = []
        for item in os.listdir(self.modelPath): # in the path
            if item.endswith(".csv") == False:  # is not csv file
                modelNames.append(item)
        return modelNames

class ModelData:
    """ Load saved a Tensorflow model and get it's paramaters """

    def __init__(self,dataPath,modelPath,modelName):
        """ Intialize ModelData Instance """
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.modelName = modelName
        
        self.historyFile = self.GetKeywordFiles("@TRAINING-HISTORY@")
        self.predictFile = self.GetKeywordFiles("@PREDICTIONS@")

        self.fullModelPath = os.path.join(self.modelPath,self.modelName)
        self.fullHistoryPath = os.path.join(self.dataPath,self.historyFile)
        self.fullPredictPath = os.path.join(self.dataPath,self.predictFile)
        
    def LoadModel(self):
        """ Load locally saved TensorFlow Model into RAM """
        modelData = keras.models.load_model(self.fullModelPath)
        weights = modelData.get_weights()
        return weights

    def LoadHistoryFile(self):
        """ Load Contents of TRAINING-HISTORY into RAM """
        historyData = pd.read_csv(self.fullHistoryPath) # load history frame
        lossData = historyData['Loss Score'].to_numpy()
        precisionData = historyData['Precision'].to_numpy()
        recallData = historyData['Recall'].to_numpy()
        return (lossData,precisionData,recallData)

    def GetKeywordFiles (self,keyWord):
        """ Collect All .csv files with 'keyWord' in it """
        for file in os.listdir(self.dataPath):  # in the path
            if file.endswith(".csv"):           # is csv file:
                if (keyWord in file) & (self.modelName in file):   # keyword in name
                    return file
        raise FileNotFoundError()
        return None

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

class BaggingAlgorithm :
    """
    Execute Baggining Algorithm
    """

    def __init__(self,modelList):
        """ Initialize BaggingAlgorithmInstance """
        self.modelList = modelList
        self.newModel = self.CloneModel()

    def CloneModel(self):
        """ Create Clone of the model to set weights to """
        parentModel = keras.models.load_model(self.modelList[0].fullModelPath)
        newModel = keras.models.clone_model(parentModel)
        for layerWeights in newModels.get_weights():
            weightShape = layerweights.shape()
            print(" ")
        return newModel

    def SetWeightsToZero(self):
        """ Set all weights in the new Model to zero """
        newWeights = self.newModel.get_weights()
        for layer in self.newModel.layers:      # each layer
            pass
            # set all weights in this layer to zero
        return self

    def WeightLosses(self):
        """ Collect & Weight the Models based on Loss """
        _delta = 1e-8               # numerical stability
        lossHistories = [] 
        lossWeighted = np.array([])
        for model in self.modelList:        # each model
            print(model.modelName)          # disp model name            
            # Collect & weight the models based on loss function?
            (loss,prec,recl) = model.LoadHistoryFile()  # load history
            lossHistories.append(loss)
            lossWeighted = np.append(lossWeighted,Mathematics.WeightLossScore(loss))
        #plotutils.TimePlotting.PlotLoss(lossHistories,"ChaoticXVal Losses")
        return 1 - (lossWeighted/np.max(lossWeighted)) + _delta

    def __Call__(self):
        """ Run Bagging Algorithm """
        lossWeights = self.WeightLosses()
        nModels = len(self.modelList)
        _ = self.newModel.get_weights()
        for i,model in enumerate(self.modelList[1:]):
            savedWeights = keras.models.load_model(model.fullModelPath)  # get a model
            modelWeights = loadedModel.get_weights()                    # get layer params

        return self

