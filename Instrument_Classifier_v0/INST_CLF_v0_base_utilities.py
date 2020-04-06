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

import scipy.io.wavfile as sciowav

"""
INSTRUMENT CLASSIFIER V0 - BASE LEVEL UTILITIES

Script contains lowest level function and class defintions
    - Supports higher end functions 
"""

            #### CLASS OBJECT DEFINITIONS ####

class wavfile():
    """
    Class : wavfile used to contain information for specific .wav file
    --------------------------------
    """
    def __init__(self,file):
        """ Initialize Class Object """
        self.filename = file                    # filename
        self.instrument = file.split('.')[0]    # Instrument name
        self.note = file.split('.')[-2]         # note name
        self.channel = file.split('.')[-1]      # L or R channel
        self.rate = 44100                       # sample rate

    def read_raw_wav(self):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw waveform
        data = np.transpose(data)               # transpose
        data = data/np.max(np.abs(data))        # normalize
        setattr(self,'waveform',data)           # set attrb to self
        return data                             # return waveform

    def split_timeseries (self,M):
        """ Split time series data into N samples by M features """
        ext = len(self.data) % M    # extra idxs
        Y = self.data[ext:]         # remove extra points
        Y = Y.reshape(-1,M)         # reshape the X matrix
        return Y                    # return cropped waveform

    def set_family(self,dict):
        """ Place each instrument into family """
        try:
            return dict[self.instrument]  # find family in dictionary
        except:
            return 'NONE'  


            #### DIRECTORY AND OS FUNCTIONS ####

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

