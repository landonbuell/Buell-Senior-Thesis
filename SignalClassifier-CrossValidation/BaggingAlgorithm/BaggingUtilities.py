"""
Landon Buell
Kevin Short
PHYS 799
18 October 2020
"""

        #### IMPORTS ####

import numpy as np
import os
import tensorflow.keras as keras

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

    def __init__(self,dtaPath,modelPath,modelName):
        """ Intialize ModelData Instance """
        self.dataPath = dataPath
        self.modelPath = modelPath
        self.modelName = modelName
        self.fullModelPath = os.path.join(self.modelPath,self.modelName)
        self.historyFile = self.GetKeyWordFiles("@TRAINING-HISTORY@")
        self.predictionFile = self.GetGetKeyWordFiles("@PREDICTIONS@")
        
    def LoadModel(self):
        """ Load locally saved ensorFlow Model into RAM """
        modelData = keras.models.load_model(self.fullModelPath)
        return modelData

    def GetKeyWordFiles (self,keyWord):
        """ Collect All .csv files with 'keyWord' in it """
        for file in os.listdir(self.dataPath):  # in the path
            if file.endswith(".csv"):           # is csv file:
                if (keyWord in file) & (self.modelName in file):   # keyword in name
                    return file


class BaggingAlgorithm :
    """
    Execute Baggining Algorithm
    """

    def __init__(self,modelList):
        """ Initialize BaggingAlgorithmInstance """
        self.modelList = modelList




