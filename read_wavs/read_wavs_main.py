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
    paths_dict = readwavs.output_paths()        # create output paths dictionary
    readwavs.make_paths(paths_dict)             # create all of the output paths

    features = (2**14)              # number of input neruons

    files = readwavs.read_directory(wav_dir)
    print("Number of 'wav' files:",len(files))

            #### READING EACH FILE ####

    for wavfile in files: 
        print("Reading File:",wavfile.filename)

        os.chdir(wavfile.dirpath)       # change to directory
        wavfile.read_raw_wav()          # read raw info
        wavfile.timespace(44100,len(wavfile.L_track))



    os.chdir(int_dir)
    print("Process time:",time.process_time())
