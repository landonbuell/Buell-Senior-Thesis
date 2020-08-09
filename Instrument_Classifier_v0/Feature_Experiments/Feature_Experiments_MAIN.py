"""
Landon Buell
Feature - Extraction
PHYS 799
6 August 2020
"""

        #### IMPORTS ####

import numpy as np
import os 
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

import System_Utilities as sys_utils
import Feature_Utilities as feat_utils
import Plotting_Utilities as plot_utils

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    path = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0\\Target-Data'
    ProgramInitializer = sys_utils.ProgramStart(path)
    FILEOBJS,N_classes = ProgramInitializer.__call__()

    #Iterator = sys_utils.FileIterator(FILEOBJS,N_classes)
    #Iterator.__call__()
    #Iterator.ExportData(str(ProgramInitializer.starttime))
    

    Analyzer = sys_utils.DataAnalyzer('2020-08-08_13.09.02.969141.csv',N_classes)
    Analyzer.__call__()
    X_train,X_test,Y_train,Y_test = Analyzer.TrainTestSplit()

    MLP_MODEL = sys_utils.NeuralNetworks.Multilayer_Perceptron('JARVIS',N_classes,Analyzer.n_features,
                                                               layerunits=[80,80])
    MLP_MODEL.fit(X_train,Y_train,batch_size=32,epochs=20,verbose=2)

    Y_pred = MLP_MODEL.predict(X_test)
    Y_pred = np.argmax(Y_pred,axis=-1)
    Y_test = np.argmax(Y_test,axis=-1)
    confmat = tf.math.confusion_matrix(Y_test,Y_pred,N_classes)

    plot_utils.Plot_Confusion(confmat,np.arange(N_classes),"-")


    print("=)")
