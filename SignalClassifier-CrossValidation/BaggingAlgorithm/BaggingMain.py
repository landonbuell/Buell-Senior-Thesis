"""
Landon Buell
Kevin Short
PHYS 799
18 October 2020
"""

        #### IMPORTS ####

import os
import sys
import BaggingUtilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # SETUP DIRECTORIES
    parentPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier-CrossValidation"
    modelName = "XValCLFB"
    dataPath = os.path.join(parentPath,"XValCLFB-Output-Data")
    modelPath = os.path.join(parentPath,"XValCLFB-Model-Data")

    # MAKE PROGRAM INTIALIZER
    Setup = utils.ProgramSetup(modelName,dataPath,modelPath)
    Setup.__Call__()

    # MAIN BAGGING ALGORITHM
    (models,training,predictions) = Setup.GetPathLists
    Bagging = utils.BaggingAlgorithm(models,training,predictions)
    Bagging.__Call__()

    print("=)")


