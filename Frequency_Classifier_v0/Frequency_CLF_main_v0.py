"""
Landon Buell
Frequency Classifer v0
Main Functions
1 January 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import time
import Frequency_CLF_func_v0 as CLF_func

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':

            #### INITIALIZING ####

    int_dir = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Frequency_Classifier_v0'
    readdir = 'C:/Users/Landon/Documents/wav_data/Frequency_Bands'
    classifiers = []                                # list to hold classifier objs
    labels = np.loadtxt('Instrument_labels.txt',
                        dtype=float,delimiter='\t',skiprows=1,usecols=-1)
    filenames = CLF_func.read_directory(readdir,'.txt')

    print("Number of Sample Labels:",len(labels))
    print("Number of files:",len(filenames))
    print("-"*32)

            #### MOVE THROUGH DATASETS ####
    os.chdir(readdir)                               # move to directory
    for file in filenames:                          # each file
        clfstart = time.process_time()
        clfname = str(file).replace('.txt','')      # clf name
        print("\nClassifier name:",clfname)           # print name
        
        xdata = CLF_func.read_csvfile(file,True)        # read dataframe
        datadict = CLF_func.random_split(xdata,labels,size=0,state=None)
        xtrain,ytrain = datadict['xtrain'],datadict['ytrain']
        classifier = CLF_func.SGD_Classifier(clfname,
                    xtrain,ytrain,state=None)
        conf_mat = CLF_func.confusion_matrix(classifier,xdata,labels,True)

        clfend = time.process_time()
        print("\tClassifier Time:",clfend-clfstart) # time to run classifier

    os.chdir(int_dir)                               # starting path
    print("Program Time:",time.process_time())      # print time
        
    
        

