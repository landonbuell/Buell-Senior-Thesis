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
    int_dir = os.getcwd()           # home path is CWD
    data_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0/INST_FTRS_v0/extdata'
    
    # LOAD DESIGN MATRIX & TARGET VECTORS
    X = pd.read_csv(data_dir+'/X.csv',header=0,index_col=0)
    y = pd.read_csv(data_dir+'/y1.csv',header=0,index_col=0)
   
    X = ML_utils.Design_Matrix_Scaler(X)
    X = ML_utils.Design_Matrix_Labeler(X,True)
    print(X.describe())

    # SPLIT TRAIN-TEST DATA
    X_train,X_test,y_train,y_test = \
        ML_utils.split_train_test(X,y,test=0.5)

    # NEURAL NETWORK
    NETWORK = ML_utils.Create_MLP_Model('All Features',(20,20),
                                        seed=None)
    NETWORK = NETWORK.fit(X_train,y_train)
    NETWORK = ML_utils.Evaluate_Classifier(NETWORK,X_test,y_test)
    #print(NETWORK.confusion)
    
    base_utils.Plot_Confusion_Matrix(NETWORK,True)
