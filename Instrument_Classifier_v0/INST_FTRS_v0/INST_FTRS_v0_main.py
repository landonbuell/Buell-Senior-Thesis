"""
Landon Buell
Instrument Classifier v0
Main Script
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import sys
import os
import time

import INST_FTRS_v0_base_utilities as base_utils
import INST_FTRS_v0_machine_learning_utilities as ML_utils
import INST_FTRS_v0_feature_utilities as feat_utils

"""
INSTRUMENT FEATURES V0 - MAIN EXECUTABLE
    Startup file for feature extraction Program
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # INITIALIZE DIRECTORIES
    int_dir = os.getcwd()           # home path is CWD
    out_dir = int_dir + '/extdata'  # path to store extra data
    base_utils.make_paths(paths=[out_dir])

    #wav_dir = base_utils.argument_parser()         # for command line 
    wav_dir = 'C:/Users/Landon/Documents/wav_audio' # for development
    if os.path.exists(wav_dir) == False:
       sys.exit("\n\tERROR - Local Path Does not Exist")
    print("Searching for .wav files in:\n\t",wav_dir)

    # COLLECT .WAV FILE INSTANCES
    WAVFILE_OBJECTS = base_utils.read_directory(wav_dir,ext='.wav')
    print("\t",len(WAVFILE_OBJECTS),"files found\n")

    # BUILD TARGET VECTOR
    y = np.array([x.target for x in WAVFILE_OBJECTS])   # target vector
    ENCODE_DICTIONARY,DECODE_DICTIONARY = \
        ML_utils.target_label_encoder(y,out_dir)        # build dict
    y = np.array([ENCODE_DICTIONARY[x] for x in y])     # convert str to int

    # BUILD DESIGN MATRIX W/ FEATURES FROM ALL FILES
    print("Constructing Design Matrix...")
    X = ML_utils.Design_Matrix(WAVFILE_OBJECTS,wav_dir,int_dir)
    X = ML_utils.Design_Matrix_Scaler(X)
    X = ML_utils.Design_Matrix_Labeler(X,True)
    
    # EXPORT X & Y
    y = pd.DataFrame(data=y,columns=['Target'])
    X.to_csv(out_dir+'/X.csv')
    y.to_csv(out_dir+'/y1.csv')
   
    
