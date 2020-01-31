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
    M = 2**12                                   # number of features

    class_dict = {}                         # dictionary to hold class values
    class_num = 0                           # class indentifier
    targets = np.array([])                  # array to hold target labels

    for wav in wavfiles:                        # for each class instance
        print("\tFilename:",wav.filename)    
        os.chdir(wav.dirpath)                   # move to the directory path
        waveform = wav.read_raw_wav()           # extract waveform
        X,N = wav.split_timeseries(M)           # split into N samples
        X = readwavs.hanning_window(X,M)        # apply hanning window to each rows`
        
        if wav.instrument not in class_dict.keys():     # if not in dictionary
            class_dict.update({str(wav.instrument):class_num})
            class_num += 1
            print("\t\tClass Counter:",class_num)

        labs = np.ones(shape=(1,N),dtype=int)*class_num     # create N labels
        targets = np.append(targets,labs)                   # add labels to tagrets
        
        outname = str(wav.filename)+'.'+str('time') # name for output file
        os.chdir(out_dir)
        X.tofile(outname+'.bin',sep="")
        del(wav)                           # remove class obj from RAM

    os.chdir(int_dir)
    targets = readwavs.to_csvfile('timeseries_targets',targets,mode='w')
    print("Process time:",time.process_time())
