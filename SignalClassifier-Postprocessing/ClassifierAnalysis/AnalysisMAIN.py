"""
Landon Buell
PHYS 799.32
Classifier Analysis Main
28 July 2020
"""

        #### IMPORTS ####

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow.keras as keras

import AnalysisUtilities as utils

        #### MAIN EXECUTABLE ####
        
if __name__ == '__main__':

    # Establish local Directory Paths
    parent = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier-CrossValidation"
    modelName = "XValCLFAlpha"
    dataPath = os.path.join(parent,modelName+"-Output-Data")
    exptPath = os.path.join(parent,modelName+"-Analysis")    
    n_classes = 38

    # Run Main program
    Program = utils.AnalyzeModels(modelName,dataPath,n_classes)
    Program.__Call__(exptPath)
    Program.ExportMetrics(exptPath)
    
    print("=)")
