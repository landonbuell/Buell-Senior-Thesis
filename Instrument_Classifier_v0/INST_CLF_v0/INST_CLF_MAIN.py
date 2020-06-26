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
    FILEOBJS = FILEOBJS[0:1000]
    print("Files Found:",len(FILEOBJS))

    # Build X & Y
    print("Contructing Design Matrix:")   
    V,W,X = ML_utils.Design_Matrices(FILEOBJS)
    Y,n_classes = ML_utils.construct_targets(FILEOBJS)

    V = V.__getmatrix__()
    W = W.__getmatrix__()
    X = X.__getmatrix__()

    # BUILD & TRAIN MULTILAYER PERCEPTRON
    print("Contructing and Training Perceptron Model...")
    n_samples,n_features = X.shape
    MLP_MODEL = NN_models.Multilayer_Perceptron('JARVIS',n_features,n_classes)
    MLP_HIST = MLP_MODEL.fit(X,Y,batch_size=64,epochs=100,verbose=2)
    plot_utils.Plot_History(MLP_HIST,MLP_MODEL)

    # BUILD & TRAIN SPECTROGRAM
    print("Constructing and Training Convolutional Model...")
    n_samples,n_rows,n_cols = W.shape
    W = W.reshape(n_samples,n_rows,n_cols,1)
    SXX_MODEL = NN_models.Convolutional_Neural_Network('VISION',(n_rows,n_cols,1),n_classes)
    SXX_HIST = SXX_MODEL.fit(W,Y,batch_size=64,epochs=100,verbose=2)
    plot_utils.Plot_History(SXX_HIST,SXX_MODEL)