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
            
    freq_bands = [('full',0,4000),('low',0,120),('midlow',120,500),
                      ('mid',500,1000),('midhigh',1000,2500),('high',2500,4000)]
    fspace = np.arange(0,22050,0.1)                 # frequency space

    csv_paths = readwavs.output_paths(freq_bands,'wav_data')    # create csv output dictionary
    fig_paths = readwavs.output_paths(freq_bands,'wav_plot')    # create figure output dictionary
    readwavs.make_paths(csv_paths)                  # create csv output paths
    readwavs.make_paths(fig_paths)                  # create figure output paths
    notefreq = readwavs.notefreq_dict()             # note to freq dictionary

    files = readwavs.read_directory(wav_dir)
    print("Number of 'wav' files:",len(files),"\n")

            #### READING EACH FILE ####
    cntr = 1
    for wavfile in files: 
        os.chdir(wavfile.dirpath)                   # change to directory
        print("Reading File:",wavfile.filename,'\t(',cntr,'of',len(files),')')     
        print("\tFilesize:",os.path.getsize(wavfile.filename))               
        wavfile.read_raw_wav()                      # read raw info
        wavfile.pitch_to_freq(notefreq)             # isolate pitch & freq

                #### TIME - SPECTRUM DATA ####

                #### FREQUENCY - SPECTRUM DATA ####   
        for channel in ['L_ch','R_ch']:         # L & R channels
            FFT = wavfile.FFT(channel)          # FFT for channel
            for band in freq_bands:             # each frequency band
                x,y,name = wavfile.freq_band(channel,FFT,fspace,band)
                os.chdir(csv_paths[str(band[0])])       # change csv output
                readwavs.to_CSV(name,[x,y],['Freq','Power'])
                os.chdir(fig_paths[str(band[0])])       # change figure output
                readwavs.Plot_Freq(x,y,name,save=True)  # plot spectrum

                #### Sprectrogram Data ####

                #### CLEANING UP ####
        del wavfile                             # delete wavefile instance
        cntr += 1
        print("\tCurrent time:",time.process_time())

    os.chdir(int_dir)
    print("Process time:",time.process_time())
