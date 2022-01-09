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
    runPath = "C:\\Users\\lando\\Documents\\audioFeatures\\devRunv0"
    runInfo = CommonStructures.RunInformation.deserialize(runPath)

    # Path to each Design Matrix
    batchIndex = 0
    designMatrices = runInfo.loadBatch(batchIndex,True,False)

    # Return 
    sys.exit(0)

