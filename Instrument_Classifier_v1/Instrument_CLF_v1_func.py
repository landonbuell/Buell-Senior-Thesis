"""
Landon Buell
Instrument Classifier v1
Main Function
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import os

import scipy.io.wavfile as sciowav

"""
INSTRUMENT CLASSIFIER V1 - MISC FUNCTIONS SCRIPT
    - wavfile class object to hold all data pertaining to particular .wav audio file
    - Walk through directory tree
    - Create missing directories
    - Reading & Writing Data
    - Time space visualization function
    - Frequnecy space visualization function
    
"""

            #### CLASS OBJECTS ####

class wavfile ():
    """
    wavfile class object to contain 
    --------------------------------
    file (str) : string to indentify file name (with ext)
    --------------------------------
    Create and return wavfile class object instance
    """

    def __init__(self,file):
        """ Initialize Class Object """
        self.filename = file                    # filename
        self.instrument = file.split('.')[0]    # Instrument name
        self.note = file.split('.')[-2]         # note name
        self.channel = file.split('.')[-1]      # L or Rchannel
        self.rate = 44100                       # sample rate

    def read_raw_wav(self):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw data
        data = np.transpose(data)               # tranpose
        data = data/np.max(np.abs(data))        # normalize
        setattr(self,'data',data)               # set attrb to self
        return data                             # return waveform

    def split_timeseries (self,M):
        """ Split time series data into N samples by M features """
        ext = len(self.data) % M    # extra idxs
        Y = self.data[ext:]         # remove extra points
        Y = np.round(Y,4)           # round to 4 decimals
        N = int(len(Y)/M)           # compute number of rows
        Y = Y.reshape(N,M)          # reshape the X matrix
        return Y,N                  # return data 

        #### READ, WRITE & ORGANIZE FUNCTIONS ####

def make_paths (paths):
    """
    Make designated directory paths
    --------------------------------
    paths (list) : List of directory paths to create
    --------------------------------
    return None
    """
    for path in paths:              # for each entry
        if os.path.exists(path):    # is the path exisits
            continue                # do nothing
        else:                       # otherwise
            os.makedirs(path)       # make the path

def read_directory(dir):
    """
    Read all files in given directory path
    --------------------------------
    dir (str) : Parent directory path to read raw .wav files from
    --------------------------------
    return list of "wav_file" class object instances
    """
    file_objs = []                          # list to '.wav' hold files objs
    for roots,dirs,files in os.walk(dir):   # all objects in parent path
        for file in files:                  # files in list of files
            if file.endswith('.wav'):       # if '.wav' file
                wavs = wavfile(file)        # make instance
                file_objs.append(wavs)      # add to list 
    return file_objs                        # return the list of files

        #### PLOTTING & VISUALIZATION FUNCTIONS #####

def Plot_Time (xdata,ydata,title,save=False,show=False):
    """
    Produce Matplotlib Figure of data in time domain
    --------------------------------
    xdata (arr) : (1 x N) Array of values for x-axis
    ydata (arr) : (M x N) Array of values for y-axis
    title (str) : Title for figure to save as
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(title+'.timeseries',size=40,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')
        #### Plotting ####
    if np.shape(ydata)[0] > 1:         
        for I in range (len(ydata)):
            plt.plot(xdata,ydata[I],color='blue')
    else:
        plt.plot(xdata,ydata)   
        #### Cleaning Up ####
    plt.hlines(0,xdata[0],xdata[-1],color='black')
    plt.yticks(np.arange(-1.0,+1.1,0.25))
    plt.xlim(xdata[0],xdata[-1])
    plt.grid()     
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.timeseries.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_Freq (xdata,ydata,title,save=False,show=False):
    """
    Produce Matplotlib Figure of data in frequency domain
    --------------------------------
    xdata (arr) : (1 x N) Array of values for x-axis
    ydata (arr) : (M x N) Array of values for y-axis
    title (str) : Title for figure to save as
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(title+'.freqseries',size=40,weight='bold')
    plt.xlabel("Frequency",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')
        #### Plotting ####
    if np.shape(ydata)[0] > 1:         
        for I in range (len(ydata)):
            plt.plot(xdata,ydata[I],color='blue')
    else:
        plt.plot(xdata,ydata)   
        #### Cleaning Up ####
    plt.hlines(0,xdata[0],xdata[-1],color='black')
    plt.yticks(np.arange(-1.0,+1.1,0.25))
    plt.xlim(xdata[0],xdata[-1])
    plt.grid()     
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.freqseries.png')
    if show == True:
        plt.show()
    plt.close()