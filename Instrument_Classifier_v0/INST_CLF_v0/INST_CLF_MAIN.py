"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import sys

import Program_Utilities as prog_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models as NN_models

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # ESTABLISH NECESSARY LOCAL PATHS
    init_path = os.getcwd()
    trgt_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0'
    itmd_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0'
    #trgt_path,itmd_path = prog_utils.Argument_Parser()
    prog_utils.Validate_Directories(trgt_path,itmd_path) 
    FILEOBJS = prog_utils.create_fileobjs(trgt_path+'/TARGETS.csv')

    # Build X & Y
    print("Contructing Design Matrix:")
    Y,n_classes = ML_utils.construct_targets(FILEOBJS)
    X = ML_utils.Design_Matrix(FILEOBJS[0:1000:100])
    n_samples,n_features = X.shape

    # BUILD NEURAL NETWORK MODELS