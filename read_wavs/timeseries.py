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

    readwavs.make_paths([out_dir])          # create the output tpath if needed
    wavfiles = readwavs.read_directory()    # create all class instances
    N = 2**10                               # number of features
    class_num = 0                           # class indentifier
    targets = np.array([])                  # array to hold target labels

    for file in wavfiles:                       # for each class instance
        os.chdir(file.dirpath)                  # move to the directory path
        waveform = file.read_raw_wav()          # extract waveform
        samples,M = file.split_timeseries(N)    # split into M samples



