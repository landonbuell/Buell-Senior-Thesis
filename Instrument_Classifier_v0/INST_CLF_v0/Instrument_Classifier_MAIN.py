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

import Component_Utilities as comp_utils

import Program_Utilities as prog_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # ESTABLISH NECESSARY LOCAL PATHS
    PATH_MAP = comp_utils.PATHS()
    PATH_MAP,FILEOBJS,Y = comp_utils.MODELS_FILEOBJS(PATH_MAP,names = ['JARVIS','VISION','ULTRON'])
    batch_step = 32      # samples/ batch
    print("Files Found:",len(FILEOBJS))

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
    