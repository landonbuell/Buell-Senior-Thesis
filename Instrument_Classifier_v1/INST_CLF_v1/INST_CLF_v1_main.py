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

import INST_CLF_v1_base_utilities as base_utils
import INST_CLF_v1_machine_learning_utilities as ML_utils
import INST_CLF_v1_feature_utilities as feat_utils

"""
INSTRUMENT CLASSIFIER v0 - MAIN EXECUTABLE
    Startup file for feature extraction Program
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # INITIALIZE DIRECTORIES
    int_dir = os.getcwd()           # home path is CWD   
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'     # for development
    out_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0' 
    #wav_dir,out_dir = base_utils.argument_parser()     # for command line
    base_utils.make_paths(paths=[out_dir])
    if os.path.exists(wav_dir) == False:
       sys.exit("\n\tERROR - Local Path Does not Exist")
    print("Searching for .wav files in:\n\t",wav_dir)

    # COLLECT .WAV FILE INSTANCES
    WAVFILE_OBJECTS = base_utils.read_directory(wav_dir,ext='.wav')
    print("\t",len(WAVFILE_OBJECTS),"files found\n")

    # BUILD TARGET VECTOR
    y = np.array([x.target for x in WAVFILE_OBJECTS])   # target vector
    ENCODE_DICTIONARY,DECODE_DICTIONARY,N_CLASSES = \
        ML_utils.target_label_encoder(y,out_dir)        # build dict
    y = np.array([ENCODE_DICTIONARY[x] for x in y])     # convert str to int

    # CREATE NETWORK CLASSIFIERS MODELS
    

    # BUILD DESIGN MATRIX W/ FEATURES FROM ALL FILES
    print("Constructing Design Matrix...")
    X = ML_utils.Design_Matrix(WAVFILE_OBJECTS,wav_dir,int_dir)

   
    
