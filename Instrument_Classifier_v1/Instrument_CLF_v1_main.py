"""
Landon Buell
Instrument Classifier v1
Main Function
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

import Instrument_CLF_v1_func as func
import Instrument_CLF_v1_features as features
import Instrument_CLF_v1_timeseries as timeseries
import Instrument_CLF_v1_freqseries as freqseries
import Instrument_CLF_v1_MLfunc as ML_func

"""
INSTRUMENT CLASSIFIER V1 - MAIN EXECUTABLE

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
    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v1'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = int_dir + '/wavdata'

    func.make_paths([out_dir])                      # create output path if non-existant
    wavfiles = func.read_directory(wav_dir)         # make all wav file instances
    tt_ratio = 0.01                                 # train/test size ratio
    trainpts,testpts = ML_func.split_train_test(len(wavfiles),tt_ratio)
    trainwavs = [wavfiles[I] for I in trainpts]     # wavs to train CLFs

    """ Read Through Each File in the TRAINING Data Set
        Produce Array of Features for each file
        Something like:
            for 'trainingfile' in 'traingwavs'
                # take single file
                # produce features matrix & labels
                # Return features & label matrix       
    """

    for wav in trainwavs:               # for each bit of training data:
        os.chdir(wav_dir)               # change to path with .wav files
        wav.read_raw_wav()              # read .wav file from path
        os.chdir(int_dir)               # move back to intial path


        



