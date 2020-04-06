"""
Landon Buell
Instrument Classifier v0
Main Script
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import time

import INST_CLF_v0_base_utilities as base_utils
import INST_CLF_v0_machine_learning_utilities as ML_utils
import INST_CLF_v0_feature_extraction as features

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

    # These paths for are Landon's Computer 
    # see documentation above to set for you particular machine

    #int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0'
    int_dir = os.getcwd()
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = int_dir + '/wavdata'

    # Initialize Program
    print("Initializing...\n")
    base_utils.make_paths(paths=[out_dir])
    WAVFILE_OBJECTS = base_utils.read_directory(wav_dir,ext='.wav')
    
    X,y = features.Xy_matrices(WAVFILE_OBJECTS)
    

    