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
            if item.endswith(".csv") == False:  # is csv file
                modelNames.append(item)
        return modelNames

class ModelData:
    """ Load saved a Tensorflow model and get it's paramaters """

    def __init__(self,savePath,modelName):
        """ Intialize ModelData Instance """
        self.savePath = savePath
        self.modelName = modelName
        self.fullPath = os.path.join(self.savePath,self.modelName)

    def LoadModel(self):
        """ Load locally saved ensorFlow Model into RAM """
        model = keras.models.load_model(self.fullPath)
        return model
