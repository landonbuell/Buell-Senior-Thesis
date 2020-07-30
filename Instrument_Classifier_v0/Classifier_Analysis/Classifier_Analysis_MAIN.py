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

    infile = 'PREDICTIONS@2020-07-29_20.43.59.585357.csv'
    model_names = ['JARVIS','VISION','ULTRON']      # names for models
    n_classes = 25

    Program_Mode = analysis_utils.Analyze_Models(model_names,export,infile,n_classes)
    Program_Mode.assign_metrics([   #keras.metrics.SparseCategoricalCrossentropy(),
                                    keras.metrics.Precision(),
                                    keras.metrics.Recall(),
                                    keras.metrics.Accuracy()    ])

    Program_Mode.read_data()
    Program_Mode.__call__()

    print("=)")
