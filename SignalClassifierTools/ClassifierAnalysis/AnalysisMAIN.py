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

    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier'
    export = os.path.join(parent,'Output-Data')

    infiles = ['JARVIS@PREDICTIONS@2020-09-21_02.41.33.904809.csv']
    n_classes = 33

    for file in infiles:
        modelName = file.split("@")[0]
        Program = utils.AnalyzeModels(modelName,export,file,n_classes)
        Program.__Call__()

    print("=)")
