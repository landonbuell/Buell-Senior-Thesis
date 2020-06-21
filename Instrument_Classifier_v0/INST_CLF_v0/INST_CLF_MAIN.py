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
    itmd_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0/Wav_Data'
    #trgt_path,itmd_path = prog_utils.Argument_Parser()
    prog_utils.Validate_Directories(trgt_path,itmd_path) 
    FILEOBJS = prog_utils.Create_Fileobjs(trgt_path+'/TARGETS.csv')

    # PROGRAM PARAMETERS
    batch_size = 1024       # samples/ batch
    FILEOBJS = np.random.permutation(FILEOBJS)
    FILEOBJS = FILEOBJS[0:10]
    print("Files Found:",len(FILEOBJS))

    # Build X & Y
    print("Contructing Design Matrix:")   
    V,W,X = ML_utils.Design_Matrices(FILEOBJS)
    Y,n_classes = ML_utils.construct_targets(FILEOBJS)
    X = X.__get_X__()
    n_samples,n_features = X.shape

    # BUILD NEURAL NETWORK MODELS
    MLP_MODEL = NN_models.Multilayer_Perceptron('JARVIS',n_features,n_classes)
    history = MLP_MODEL.fit(X,Y,batch_size=64,epochs=10,verbose=2)
    plot_utils.Plot_History(history,MLP_MODEL)