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
    readdir = 'C:/Users/Landon/Documents/wav_data/frequencies'
    paths = [roots.replace('\\','/') for roots,dirs,files in os.walk(readdir)][2:]
    classifiers = []                                # list to hold classifier objs
    labels = np.loadtxt('Instrument_labels.txt',
                        dtype=int,delimiter='\t',skiprows=1,usecols=-1)

            #### MOVE THROUGH DATASETS ####
        
    for dataset in paths:                           # for each directory
        clfname = dataset.split('/')[-1]            # name for classifier
        print("SGD Classifier Name:",clfname)       # print out name
        
        os.chdir(dataset)                                   # move to specific directory
        filenames = CLF_func.read_directory(dataset,'.txt') # names of 
        xdata = CLF_func.assemble_dataset(filenames)        # build X matrix
        print(np.shape(xdata))
        A = input("wait:")
        

