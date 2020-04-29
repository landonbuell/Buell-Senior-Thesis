"""
Landon Buell
Instrument Classifier v0
Base Level Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import argparse
import matplotlib.pyplot as plt

import scipy.io.wavfile as sciowav

"""
INSTRUMENT CLASSIFIER V0 - BASE LEVEL UTILITIES

Script contains lowest level function and class defintions
    - Supports higher end functions 
"""

            #### VARIABLE & OBJECT DECLARATIONS ####   

INSTRUMENT_FAMILIES = {}

            #### CLASS OBJECT DEFINITIONS ####

class wavfile():
    """
    Class : wavfile used to contain information for specific .wav file
    --------------------------------
    file (str) : name of file (with extention) in local path
    --------------------------------
    Returns initialized class instance
    """
    def __init__(self,file):
        """ Initialize Class Object """
        self.filename = file                    # filename
        self.instrument = file.split('.')[0]    # Instrument name
        self.note = file.split('.')[-2]         # note name
        self.channel = file.split('.')[-1]      # L or R channel
        self.rate = 44100                       # sample rate

    def set_target (self,target):
        """ Set integer target value based on instrument """
        self.target = target
        return self

    def read_raw_wav(self,normalize=True):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw waveform
        data = np.transpose(data)               # transpose
        if normalize == True:
            data = data/np.max(np.abs(data))    # normalize
        self.waveform = data                    # attach waveform
        self.n_pts = len(data)                  # attach num samples
        return self                             # return instance

            #### DIRECTORY AND OS FUNCTIONS ####

def argument_parser():
    """
    Create argument parper object for main executable
    --------------------------------
    *no argumnets*
    --------------------------------
    Return completetd argument parser class instance
    """
    parser = argparse.ArgumentParser(prog='Instrument Classifier v0')
    parser.add_argument('wav_audio',type=str,
                        help='Local path where raw audio is stored')
    args = parser.parse_args()
    return args.wav_audio 

def make_paths(paths=[]):
    """
    Create directory paths for each
    --------------------------------
    paths (iter) : iterable of folder directory paths to make
    --------------------------------
    returns NONE
    """
    for path in paths:
        os.makedirs(path,exist_ok=True)
    return paths

def read_directory (path,ext='.wav'):
    """
    Read through directory and create instance of every file with matching extension
    --------------------------------
    path (str) : file path to read data from
    ext (str) : extension for appropriate file types
    --------------------------------
    Return list of wavfile class instances
    """
    file_objs = []                              # list to hold valid file
    for roots,dirs,files in os.walk(path):      # walk through the tree
        for file in files:                      # for each file
            if file.endswith(ext):              # matching extension
                file_objs.append(wavfile(file)) # add instance to list 
    return file_objs                            # return list of instances

            #### MECHANICAL FUNCTIONS ###

def normalize_X (X):
    """
    Normalize design matrix by each feature column
    --------------------------------
    X (arr) : (n_samples x n_features) array to normalize
    --------------------------------
    Return X normalized by column
    """
    for row in X.transpose():       # each feature
        row /= np.max(np.abs(row))  # normalize
    return X                        # return new matrix

            #### PLOTTING FUNCTIONS ####

def Plot_Time_Spectrum (xdata,ydata,labels,title='',show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    xdata (arr) : (1 x N) array of data to plot on x-axis
    ydata (arr) : (M x N) array of data to plot on y-axis ( can be multiple arrays)
    labels (iter) : (1 x 2) iterable containing labels for x & y axes
    title (str) : Title for plot
    --------------------------------
    return None
    """
    plt.figure(figsize=(12,8))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Time',size=20,weight='bold')
    plt.ylabel('Amplitude',size=20,weight='bold')

    if ydata.ndim > 1:
        for I in len(ydata):
            plt.plot(xdata,ydata[I],label=str(labels[I]))
        plt.legend()

    else:
        plt.plot(xdata,ydata)

    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    #plt.hlines(0,0,xdata[-1],color='black')
    
    plt.grid()
    plt.tight_layout()
    if show == True:
        plt.show()

def Plot_Freq_Spectrum (xdata,ydata,labels,title='',show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    xdata (arr) : (1 x N) array of data to plot on x-axis
    ydata (arr) : (M x N) array of data to plot on y-axis ( can be multiple arrays)
    labels (iter) : (1 x 2) iterable containing labels for x & y axes
    title (str) : Title for plot
    --------------------------------
    return None
    """
    plt.figure(figsize=(12,8))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Frequency',size=20,weight='bold')
    plt.ylabel('Amplitude',size=20,weight='bold')

    if ydata.ndim > 1:
        for I in len(ydata):
            plt.plot(xdata,ydata[I],label=str(labels[I]))
        plt.legend()

    else:
        plt.plot(xdata,ydata)

    plt.yticks(np.arange(0,1.1,0.1))
    plt.hlines(0,0,xdata[-1],color='black')
    
    plt.grid()
    plt.tight_layout()
    if show == True:
        plt.show()


def Plot_Spectrogram (t,f,spect,title,show=True):
    """
    Create visualization of soundwave as frequency vs. time vs. power
    --------------------------------
    spect ((arr) : (N x M) matrix representing file's spectrogram
    title (str) : Title for plot
    --------------------------------
    return None
    """
    plt.figure(figsize=(12,8))
    plt.title(title,size=40,weight='bold')
    plt.xlabel('Time',size=20,weight='bold')
    plt.ylabel('Frequnecy',size=20,weight='bold')

    plt.pcolormesh(t,f,spect,cmap=plt.cm.binary)

    plt.grid()
    plt.tight_layout()
    if show == True:
        plt.show()



def Plot_Features_2D (X1,X2,classes,labels,title='',show=True):
    """
    Create 2D visualization Comparing features
    --------------------------------
    X1 (arr) : (1 x N) array of data to plot on x-axis
    X2 (arr) : (1 x N) array of data to plot on y-axis
    classes (arr) : (1 x N) array of labels use to color-code by class
    labels (iter) : (1 x 2) iterable containing labels for x & y axes
    title (str) : Title for plot
    --------------------------------
    return None
    """
    plt.figure(figsize=(16,12))
    plt.title(title,size=40,weight='bold')
    plt.xlabel(str(labels[0]),size=20,weight='bold')
    plt.ylabel(str(labels[1]),size=20,weight='bold')

    plt.scatter(X1,X2,c=classes)

    plt.yticks(np.arange(0,1.1,0.1))
    plt.hlines(0,0,1,color='black')
    plt.vlines(0,0,1,color='black')

    plt.grid()
    plt.tight_layout()
    if show == True:
        plt.show()

