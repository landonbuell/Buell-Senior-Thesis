"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           __main__.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys

import Preprocessing
import CommonStructures

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Parse User Arguments
    runPath = "C:\\Users\\lando\\Documents\\audioFeatures\\devRunv1"
    runInfo = CommonStructures.RunInformation.deserialize(runPath)

    # Path to each Design Matrix
    batchIndex = 255
    designMatrices = runInfo.loadAllSamples(True,False,)
    
    designMatrixA = designMatrices[0]
    designMatrixA.dropNaNs()

    scaler = Preprocessing.FeatureScaler()
    scaler.fitDesignMatrix(designMatrixA)
    designMatrixA = scaler.transform(designMatrixA)
    

    # Return 
    sys.exit(0)

