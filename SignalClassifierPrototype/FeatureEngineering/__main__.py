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
    runPath = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV2"
    runInfo = CommonStructures.RunInformation.deserialize(runPath)

    # Path to each Design Matrix
    designMatrices = runInfo.loadAllSamples(True,False)
    matrixA = designMatrices[0].dropNaNsAndInfs()
    
    tool = Preprocessing.MinMaxVarianceSelector()
    tool.fit(matrixA)




    # Return 
    sys.exit(0)

