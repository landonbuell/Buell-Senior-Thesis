"""
Landon Buell
Wav file to Frequency Bands
PHYS 799
28 December 2019
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import read_wavs_func as readwavs

if __name__ == '__main__':

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/read_wavs'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = 'C:/Users/Landon/Documents/wav_data/time_series'

    readwavs.make_paths([out_dir])              # create the output tpath if needed
    wavfiles = readwavs.read_directory(wav_dir) # create all class instances
    print("Number of .wav files:",len(wavfiles))
    N = 2**10                                   # number of features

    class_dict = {}                         # dictionary to hold class values
    class_num = 0                           # class indentifier
    targets = np.array([])                  # array to hold target labels

    for file in wavfiles:                       # for each class instance
        print("\tFilename:",file.filename)    
        os.chdir(file.dirpath)                  # move to the directory path
        waveform = file.read_raw_wav()          # extract waveform
        X,M = file.split_timeseries(N)          # split into M samples
        X = readwavs.hanning_window(X,N)        # apply hanning window to each rows`
        
        if file.instrument not in class_dict:           # if not in dictionary
            class_dict.update({str(file.instrument):class_num})
            class_num += 1

        """
        To impliment:
        Each instrument gets it's own .txt file
        The target data array will stil include all sample labels in a single vector
        """

        labs = np.arange(0,M,1)*class_num       # create M labels
        targets = np.append(targets,labs)       # add labels to tagrets

        os.chdir(out_dir)

        del(file)                           # remove class obj from RAM

    os.chdir(int_dir)
    targets = readwavs.to_csvfile('timeseries_targets',targets,mode='w')
    print("Process time:",time.process_time())
