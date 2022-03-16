"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        NeuralNetworkModels
File:           __main__.py
 
Author:         Landon Buell
Date:           January 2022
"""

    #### IMPORTS ####

import os
import sys

import Managers

import NeuralNetworkModels

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Build the Tensor flow Model
    modelManager = Managers.TensorflowModelManager()
    hybridNetwork = modelManager.generateModel()
    
    # Load in the Data Set
    batchIndex = 12
    dataManager = Managers.DatasetManager()
    dataManager.loadBatch(batchIndex)



    sys.exit(0)