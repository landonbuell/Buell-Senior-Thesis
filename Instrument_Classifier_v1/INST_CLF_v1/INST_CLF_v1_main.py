"""
Landon Buell
Instrument Classifier v1
Classifier - Neural Network Models
4 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import sys
import os
import time

import INST_CLF_v1__Machine_Learning_Utilities as ML_utils
import INST_CLF_v1_Network_Models as NN_Models
import INST_CLF_v1_base_utilities as base_utils

"""
INSTRUMENT CLASSIFIER v0 - MAIN EXECUTABLE
    Startup file for feature extraction Program
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # INITIALIZE DIRECTORIES
    int_dir = os.getcwd()           # home path is CWD   
    wav_dir = 'C:/Users/Landon/Documents/wav_data'     # for development
    #wav_dir = base_utils.argument_parser()     # for command line
    if os.path.exists(wav_dir) == False:
       sys.exit("\n\tERROR - Local Path Does not Exist")
    print("Searching for .wav files in:\n\t",wav_dir)
 
    # IMPORT X,y 
    DECODE = pd.read_csv(wav_dir+'/DECODE.csv',index_col=0)
    X = pd.read_csv(wav_dir+'/X.csv',index_col=0)
    y = pd.read_csv(wav_dir+'/y.csv',index_col=0).to_numpy()

    # PRE-PROCESSING
    X = ML_utils.Scale_Design_Matrix(X) 
    X_train,X_test,y_train,y_test = \
        ML_utils.split_train_test(X,y,test=0.4)
    Y_train,n_classes = ML_utils.one_hot_encoder(y_train)
    n_samples,n_features = X.shape

    MLP = False
    if MLP == True:
        # MLP MODEL
        MLP_MODEL = NN_Models.MLP_Classifier('JARVIS',n_features,n_classes,
                                             path=int_dir,metrics=['Precision','Recall'])
        MLP_MODEL.fit(x=X_train,y=Y_train,batch_size=128,epochs=200,verbose=2)
        MLP_MODEL = ML_utils.Evaluate_Model(MLP_MODEL,X_test,y_test)
        MLP_MODEL.save(int_dir,save_format='tf')

    CNN = True
    if CNN == True:
        CNN_MODEL = NN_Models.CNN_2D_Classifier('VISION',n_classes,
                                                path=int_dir,metrics=['Precision','Recall'])
