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

import CommonStructures

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Parse User Arguments
    runPath = "C:\\Users\\lando\\Documents\\audioFeatures\\devTestv0"
    runInfo = CommonStructures.RunInformation.deserialize(runPath)

    # Path to each Design Matrix
    batchIndex = 0
    runData = runInfo.loadAllSamples()
    designMatrixA = runData[0]
    designMatrixB = runData[1]
    
    # Check Means + Variances
    means = designMatrixA.averageOfFeatures()
    varis = designMatrixA.varianceOfFeatures()

    # Return 
    sys.exit(0)

