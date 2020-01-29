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

            #### INITIALIZING ####

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/read_wavs'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    csv_dir = 'C:/Users/Landon/Documents/wav_data/Frequency_Bands'
    
    readwavs.make_paths([csv_dir])
    rate = 44100

            #### BUILDING FREQUENCY BANDS ####
    print("Building Frequency Bands...")
    freq_bands = [('band1',0,200),('band2',200,500),('band3',500,1000),('band4',1000,1500),
                  ('band5',1500,2000),('band6',2000,2500),('band7',2500,3000),('band8',3000,3500),
                  ('band9',3500,4000)]
    fspace = np.arange(0,rate/2,0.1)                                # frequency space
    os.chdir(csv_dir)                                               # change to directory
    for band in freq_bands:                                         # each frequency band
        print("\t"+band[0]+"...")
        freq_pts = np.where((fspace>=band[1])&(fspace<=band[2]))    # isolate pts in frequency space
        freq_pts = fspace[freq_pts].round(4)                        # frequency band
        outfile = band[0]+'_'+str(band[1])+'-'+str(band[2])         # name for CSV file       
        readwavs.to_csvfile(outfile,freq_pts,['Frequency'],mode='w')# write pts to dataframe

            #### PASSING THROUGH EACH FILE ####
    files = readwavs.read_directory(wav_dir)            # find each wav file
    print("Number of 'wav' files:",len(files),"\n")     # number of files
        
    for wavfile in files:                               # for Each file:
        os.chdir(wavfile.dirpath)                       # change to directory
        print("\tFile:",wavfile.filename)               # print filename
        wavfile.read_raw_wav()                          # read the file
        #wavfile.pitch_to_freq(notefreq)                 # find note frequency                 

        data_FFT = wavfile.FFT('data')              # compute FFT for file
        os.chdir(csv_dir)
        for band in freq_bands:                     # in Each F_band
            freq_pts = np.where((fspace>=band[1])&(fspace<=band[2]))    # isolate pts in frequency space
            power = data_FFT[freq_pts]                                  # slcie FFT          
            outfile = band[0]+'_'+str(band[1])+'-'+str(band[2])         # name for CSV file       
            readwavs.to_csvfile(outfile,data=power,
                        labels=[wavfile.filename],mode='a')             # write row to dataframe
        del wavfile                                 # delete class instance
    
    os.chdir(int_dir)       # starting directory
    print("Process time:",time.process_time())