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
import Neural_Network_Models

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # ESTABLISH NECESSARY LOCAL PATHS
    init_path = os.getcwd()
    trgt_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0'
    save_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0/Model-Data'
    #trgt_path,itmd_path = prog_utils.Argument_Parser()
    prog_utils.Validate_Directories(trgt_path,save_path) 
    FILEOBJS = prog_utils.Create_Fileobjs(trgt_path+'/TARGETS.csv')

    # PROGRAM PARAMETERS
    batch_step = 256      # samples/ batch
    FILEOBJS = np.random.permutation(FILEOBJS)
    Y,n_classes = ML_utils.construct_targets(FILEOBJS)
    print("Files Found:",len(FILEOBJS))

    # CONSTRUCT NEURAL NETWORKS MODELS

    MLP = Neural_Network_Models.Multilayer_Perceptron('JARVIS',
            n_features=15,n_classes=n_classes,layer_units=(20,))
    MLP.save(filepath=save_path+'/'+str(MLP.name),overwrite=True)
    CNN = Neural_Network_Models.Convolutional_Neural_Network('VISION',
            in_shape=Neural_Network_Models.Sxx_shape,n_classes=n_classes,layer_units=(20,))
    CNN.save(filepath=save_path+'/'+str(MLP.name),overwrite=True)


    # ITERATE THROUGH DATA SET BY BATCH
    for I in range(0,len(FILEOBJS),batch_step):          
        BATCH = FILEOBJS[I:I+batch_step]     
        V,W,X = ML_utils.Design_Matrices(BATCH)


    