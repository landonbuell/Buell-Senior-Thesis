"""
Landon Buell
Instrument Classifier v0
Main Script
10 May 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils


"""
INSTRUMENT CLASSIFIER V0 - MAIN EXECUTABLE
    Startup file for Instrument Classification Program
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # INITIALIZE DIRECTORIES
    int_dir = os.getcwd()           # home path is CWD\
    #data_dir = base_utils.argument_parser()         # for command line 
    data_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/'+ \
                'Instrument_Classifier_v0/INST_FTRS_v0/extdata'
    
    # LOAD DESIGN MATRIX & TARGET VECTORS
    X = pd.read_csv(data_dir+'/X.csv',header=0,index_col=0)
    y = pd.read_csv(data_dir+'/y1.csv',header=0,index_col=0)
    
    n_samples,n_features = X.shape

    DECODE_DICTIONARY = ML_utils.target_label_decoder(data_dir)
    print(DECODE_DICTIONARY)
   
    X = ML_utils.Design_Matrix_Scaler(X)
    X = ML_utils.Design_Matrix_Labeler(X,True)
    print(X.describe())

    # SPLIT TRAIN-TEST DATA
    X_train,X_test,y_train,y_test = \
        ML_utils.split_train_test(X,y,test=0.3)
    Y_train,n_classes = ML_utils.One_Hot_Encoder(y_train)

    # NEURAL NETWORK
    NETWORK = ML_utils.Create_Sequential_Model('JARVIS',n_features,n_classes)
    NETWORK.fit(x=X_train,y=Y_train,batch_size=32,epochs=500)

    NETWORK = ML_utils.Evaluate_Classifier(NETWORK,X_test,y_test)

    #print(NETWORK.confusion)
    base_utils.Plot_Confusion_Matrix(NETWORK,
        np.arange(0,n_classes,1),True)
