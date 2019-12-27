"""
Landon Buell
Instrument Classifier  v0
Main Executable Function
27 December 2019
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os
import inst_clf_func_v0 as func_v0
import inst_clf_comp_v0 as comp_v0

        #### MAIN EXECUTABLE ####

if __name__ == '__main__':

    int_dir = os.getcwd()                               # starting directory path
    read_dir = 'C:/Users/Landon/Documents/wav_audio'    # location of raw wavs

    wavs = func_v0.read_directory(readdir)              # find all wavs
    print("Number of files to read in this path:",len(wavs))


