"""
Landon Buell
Build Formant Structures
Main
2 Feb 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import SGDClassifier

import formants_func_v0 as formant

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Formant_Classifier_v0'
    wav_dir = 'C:/Users/Landon/Documents/wav_audio'
    out_dir = 'C:/Users/Landon/Documents/wav_data/formants'

    formant.make_paths([out_dir])               # create the output tpath if needed
    wavfiles = formant.read_directory(wav_dir)  # create all class instances
    print("Number of .wav files:",len(wavfiles))
    M = 2**12                                   # number of features

    class_dict = {}                         # dictionary to hold class values
    class_num = 0                           # class indentifier
    targets = np.array([])                  # array to hold target labels

    for wav in wavfiles[0:72:1]:             # for each class instance

        if wav.instrument not in class_dict.keys():     # if not in dictionary
            class_dict.update({str(wav.instrument):class_num})
            class_num += 1
            flag = True

        print("\tFilename:",wav.filename)    
        os.chdir(wav.dirpath)                   # move to the directory path
        waveform = wav.read_raw_wav()           # extract waveform
        X,N = wav.split_timeseries(waveform,M)  # split into N samples
        
        axis,pts = formant.Frequency_Space()    # create frequency axis
        F = formant.FFT_Matrix(X,pts)
        
        for row in F:
            plt.plot(axis,row)
   
        #del(wav)                           # remove class obj from RAM

    plt.show()


    os.chdir(int_dir)
    


