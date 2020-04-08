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

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils
import INST_CLF_v0_feature_utilities as feat_utils

"""
INSTRUMENT CLASSIFIER V0 - MAIN EXECUTABLE

Directory paths:
    - 'int_dir' is the initial directory for this program
            where it is saved locally on the HDD or SSD
            also use "int_dir = os.getcwd()" as well
    - 'wav_dir' is the directory path where all raw .wav audio files are stored
    - 'out_dir' is a misc directory use to dump temporary files (created by program if nonexisitant)
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    # INITIALIZE DIRECTORIES
    int_dir = os.getcwd()           # home path is CWD
    out_dir = int_dir + '/wavdata'  # path to store extra data
    base_utils.make_paths(paths=[out_dir])

    #wav_dir = base_utils.argument_parser()          # for command line 
    wav_dir = 'C:/Users/Landon/Documents/wav_audio' # for development
    if os.path.exists(wav_dir) == False:
       sys.exit()
    

    # COLLECT .WAV FILE INSTANCES
    WAVFILE_OBJECTS = base_utils.read_directory(wav_dir,ext='.wav')
    print("Number of Files Found:",len(WAVFILE_OBJECTS))

    # COLLECT FEATURES FROM ALL FILES
    X,y = ML_utils.Xy_matrices(WAVFILE_OBJECTS,wav_dir,int_dir)
    

    