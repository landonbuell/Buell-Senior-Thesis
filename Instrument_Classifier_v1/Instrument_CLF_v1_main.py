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
import Instrument_CLF_v1_timeseries as timeseries
import Instrument_CLF_v1_freqseries as freqseries
import Instrument_CLF_v1_MLfunc as ML_func

"""
INSTRUMENT CLASSIFIER V1 - MAIN EXECUTABLE

Directory paths:
    - 'int_dir' is the initial directory for this program - usually when it is saved locally on the HDD or SSD
    - 'wav_dir' is the directory path where all raw .wav audio files are stored
    - 'out_dir' is a misc directory use to dump temporary files (created by program if nonexisitant)
"""

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v1'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = 'C:/Users/Landon/Documents/wav_data'

    func.make_paths([out_dir])                      # create output path if non-existant
    wavfiles = func.read_directory(wav_dir)         # make all wav file instances
    tt_ratio = 0.01                                 # train/test size ratio
    trainpts,testpts = ML_func.split_train_test(len(wavfiles),tt_ratio)
    trainwavs = [wavfiles[I] for I in trainpts]     # wavs to train CLFs





