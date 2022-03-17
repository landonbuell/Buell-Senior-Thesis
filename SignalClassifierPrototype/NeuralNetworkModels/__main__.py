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

import numpy as np

import Managers

import NeuralNetworkModels

    #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Build the Tensor flow Model
    modelManager = Managers.TensorflowModelManager()
    hybridNetwork = modelManager.generateModel()
    
    # Load + Preprocess the Data Set 
    batchesToLoad = np.arange(0,274,1)
    dataManager = Managers.DatasetManager(batchesToLoad)  
    dataManager.preprocessSamples()



    # Scale the Data Set


    sys.exit(0)