"""
Landon Buell
PHYS 799
Read wav files - main
28 December 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import read_wavs_func as readwavs


if __name__ == '__main__':

            #### INITIALIZING ####

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/read_wavs'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    features = (2**14)              # number of input neruons

    files = readwavs.read_directory(wav_dir)
    print("Number of 'wav' files:",len(files))

            #### READING EACH FILE ####

    for wavfile in files: 
        print("Reading File:",wavfile.filename)

        os.chdir(wavfile.dirpath)       # change to directory
        wavfile.read_raw_wav()          # read raw info
        wavfile.timespace(44100,len(wavfile.L_track))

        readwavs.Plot_Time(wavfile,wavfile.filename,['L_track'],show=True)

