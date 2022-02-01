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

import Adminstrative
import Preprocessing
import CommonStructures

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # Parse User Arguments
    runPath = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV2"
    outPath = "C:\\Users\\lando\\Documents\\audioFeatures\\simpleSignalsV2Cleaned"
    runInfo = CommonStructures.RunInformation.deserialize(runPath)

    # Set Up the Processing pipeline
    queue = Adminstrative.PreprocessingQueue()
    queue.enqueue(Preprocessing.FeatureScaler())
    
    # Proceess Full Output
    queue.processCollectionRun(runInfo,outPath)
   

    # Return 
    sys.exit(0)

