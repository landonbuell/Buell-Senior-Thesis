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

import Analysis_Utilities as analysis_utils

        #### MAIN EXECUTABLE ####
        
if __name__ == '__main__':

    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    export = os.path.join(parent,'Output-Data')

    infile = 'VISION@PREDICTIONS@2020-08-27_13.20.24.810306.csv'
    modelName = infile.split("@")[0]
    n_classes = 25

    Program = analysis_utils.Analyze_Models(modelName,export,infile,n_classes)
    Program.__Call__()

    print("=)")
