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
import Instrument_CLF_v1_MLfunc as MLfunc

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
    MLfunc.label_encoder(wavfiles)                  # make numerical labels
    tt_ratio = 0.01                                 # train/test size ratio
    trainpts,testpts = MLfunc.split_train_test(len(wavfiles),tt_ratio)
    
    trainwavs = [wavfiles[I] for I in trainpts]     # wavs to train CLFs
    testwavs = [wavfiles[I] for I in testpts]       # wavs to test CLFs
    os.chdir(wav_dir)           # change to wav directory
    for wav in trainwavs:       # each training file
        wav.read_raw_wav()      # read waveform
    os.chdir(int_dir)           # intial dir
    SGD_CLF_dict = MLfunc.SGD_CLFs(['time_clf','freq_clf',
                                    'form_clf','spect_clf'])

    """ Train Time-Domain Classifier On Each .wav file """ 
    print("Training Time Series:")
    features.time_domain_features(trainwavs,SGD_CLF_dict['time_clf'])

    print(time.process_time())



