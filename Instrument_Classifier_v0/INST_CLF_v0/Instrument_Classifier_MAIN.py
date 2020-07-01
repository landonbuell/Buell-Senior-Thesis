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
    FILEOBJS = np.random.permutation(FILEOBJS)
    Y,n_classes = ML_utils.construct_targets(FILEOBJS)
    batch_step = 32      # samples/ batch
    print("Files Found:",len(FILEOBJS))

    # CONSTRUCT NEURAL NETWORKS MODELS
    names = ['JARVIS','VISION','ULTRON']
    MODEL = Neural_Network_Models.Multilayer_Perceptron(names[0],
                n_features=15,n_classes=n_classes,layerunits=[40,40])
    MLP_path = os.path.join(save_path,names[0])
    MODEL.save(MLP_path)

    MODEL = Neural_Network_Models.Convolutional_Neural_Network_2D(names[1],
                in_shape=(Neural_Network_Models.sepectrogram_shape),n_classes=n_classes,
                kernelsizes=(3,3),layerunits=[128])
    CNN_path = os.path.join(save_path,names[1])
    MODEL.save(CNN_path)

    # ITERATE THROUGH DATA SET BY BATCH
    for I in range(0,len(FILEOBJS),batch_step):          
        BATCH = FILEOBJS[I:I+batch_step]        # file in batch
        TRGTS = Y[I:I+batch_step]               # target vector
        V,W,X = ML_utils.Design_Matrices(BATCH)

        # DATA FOR PHASE-SPACE

        # DATA FOR SPECTROGRAM
        W = W.__getmatrix__()
        MODEL = Neural_Network_Models.keras.models.load_model(CNN_path)
        MODEL.fit(x=W,y=TRGTS,batch_size=16,epochs=20,verbose=2)
        MODEL.save(CNN_path)

        # DATA FOR MLP
        X = X.__getmatrix__()
        MODEL = Neural_Network_Models.keras.models.load_model(MLP_path)
        MODEL.fit(x=X,y=TRGTS,batch_size=16,epochs=20,verbose=2)
        MODEL.save(MLP_path)


    