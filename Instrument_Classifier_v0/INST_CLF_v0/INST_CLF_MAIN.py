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
    batch_step = 8     # samples/ batch
    FILEOBJS = np.random.permutation(FILEOBJS)
    FILEOBJS = FILEOBJS[0:1024]
    print("Files Found:",len(FILEOBJS))

    # Build Netwotk Models
    MLPname = 'JARVIS'
    MLP_MODEL = NN_models.Multilayer_Perceptron(MLPname,15,n_classes=25,layerunits=[40,40],
                                                metrics=['Precision','Recall'])
    MLPpath = os.path.join(itmd_path,MLPname)
    MLP_MODEL.save(MLPpath,overwrite=True)
    
    CNNname = 'VISION'
    CNN_MODEL = NN_models.Convolutional_Neural_Network(CNNname,in_shape=NN_models.sepectrogram_shape,
                       n_classes=25,kernelsizes=(3,3),layerunits=[128],metrics=['Precision','Recall'])
    CNNpath = os.path.join(itmd_path,CNNname)
    CNN_MODEL.save(CNNpath,overwrite=True)

    # Build X & Y
    print("Contructing Design Matrix:") 
    for I in range (0,len(FILEOBJS),batch_step):
        print("\tBatch Slice Size:",batch_step)
        FILEOBJ_BATCH = FILEOBJS[I:I+batch_step]
        V,W,X = ML_utils.Design_Matrices(FILEOBJ_BATCH)
        Y,n_classes = ML_utils.construct_targets(FILEOBJ_BATCH)

        # DATA FOR PHASE-SPACE
        V = V.__getmatrix__()

        # DATA FOR SPECTROGRAM
        W = W.__getmatrix__()
        MODEL = NN_models.keras.models.load_model(CNNpath)
        MODEL.fit(x=W,y=Y,batch_size=8,epochs=20,verbose=2)
        MODEL.save(CNNpath,overwrite=True)

        # DATA FOR MLP
        X = X.__getmatrix__()
        MODEL = NN_models.keras.models.load_model(MLPpath)
        MODEL.fit(x=X,y=Y,batch_size=8,epochs=20,verbose=2)
        MODEL.save(MLPpath,overwrite=True)




        