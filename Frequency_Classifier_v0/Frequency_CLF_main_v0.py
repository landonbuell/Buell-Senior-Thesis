"""
Landon Buell
Frequency Classifer v0
Main Functions
1 January 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import Frequency_CLF_func_v0 as CLF_func

if __name__ == '__main__':

            #### INITIALIZING ####

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Frequency_Classifier_v0'
    readdir = 'C:/Users/Landon/Documents/wav_data/Frequency_Bands'
    classifiers = []                                # list to hold classifier objs
    labels = np.loadtxt('Instrument_labels_double.txt',
                        dtype=int,delimiter='\t',skiprows=1,usecols=-1)
    filenames = CLF_func.read_directory(readdir,'.txt')
    print("Number of ")

            #### MOVE THROUGH DATASETS ####
    os.chdir(readdir)                                       # move to directory
    for file in filenames:                                  # each file
        clfname = str(file).replace('.txt','')              # clf name
        print("Classifier name:",clfname)                   # print name
        xdata = np.loadtxt(file,dtype=str,delimiter='\t',
                           skiprows=1,unpack=True)          # read data from CSV
        filenames = xdata[0]                                # instrument names
        xdata = np.array(xdata[1:],dtype=float)             # xdata 

        
    
        

