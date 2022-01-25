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
    batches = range(0,runInfo.getNumBatches(),3)
    designMatrices = runInfo.loadBatches(batches,True,False)

    matrixA = designMatrices[0].dropNaNs()
    
    tool = Preprocessing.MinMaxVarianceSelector()
    tool.fit(matrixA)




    # Return 
    sys.exit(0)

