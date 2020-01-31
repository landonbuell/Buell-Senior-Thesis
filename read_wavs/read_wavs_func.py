"""
Landon Buell
Read Raw .wav files
Functions
28 December 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import scipy.signal as signal
import scipy.io.wavfile as sciowav
import scipy.fftpack as fftpack

        #### CLASS OBJECTS ####

class wav_file ():
    """ Create Raw wavefile object """

    def __init__(self,root,file):
        """ Initialize Class Object """
        self.dirpath = root                     # intial storage path
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
        ext = len(self.data) % M        # extra idxs
        X = self.data[ext:]             # remove extra points
        X = np.round(X,4)               # round to 4 decimals
        N = int(len(X)/M)               # compute number of rows
        X = X.reshape(N,M)              # reshape the X matrix
        return X,N                      # return data 



        #### FUNCTION DEFINITIONS ####

def hanning_window (data,M):
    """ 
    Apply a Hanning Window Taper to each sample 
    --------------------------------
    data (array) : N x M array of floating point time series data
    M (int) : Number of rows of matrix, length of Hanning Window
    --------------------------------
    Return a matrix with a Hanning taper applied to each row
    """
    taper = signal.hanning(M=M,sym=True)        # hanning window Taper
    for x in data:                              # for each row
        x *= taper                              # apply window
    return data                                 # return new matrix


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
                wavs = wav_file(roots,file) # make instance
                file_objs.append(wavs)      # add to list 
    return file_objs                        # return the list of files

def to_csvfile (name,data,mode='w'):
    """ Append pos & vel arrays to end of csv """
    frame = pd.DataFrame(data=data,dtype=float)
    frame.to_csv(name+'.txt',sep='\t',
                    header=False,index=False,mode=mode)     # append to CSV 
    return frame

        #### PLOTTING & VISUALIZATION FUNCTIONS #####

def Plot_Time (xdata,ydata,title,save=False,show=False):
    """
    Produce Matplotlib Figure of data in time domain
    --------------------------------
    xdata (arr) : Array of values for x-axis
    ydata (arr) : Array of values for y-axis
    title (str) : Title for figure to save as
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(title,size=40,weight='bold')
    plt.xlabel("Time",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')

    plt.plot(xdata,ydata,color='blue')
        
    plt.hlines(0,xdata[0],xdata[-1],color='black')
    plt.yticks(np.arange(-1.0,+1.1,0.25))
    plt.xlim(xdata[0],xdata[-1])
    plt.grid()
        
    plt.hlines(0,obj.time[0],obj.time[-1],color='black')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.time.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_Freq (xdata,ydata,title,save=False,show=False):
    """
    Produce Matplotlib Figure of data in Frequency Domain
    Note: Must have executed FFT & Freq_Space methods before calling
    --------------------------------
    xdata (arr) : Array of values for x-axis
    ydata (arr) : Array of values for y-axis
    title (str) : Title for figure to save as
    save (bool) : indicates to progam to save figure to cwd (False by default)
    show (bool) : indicates to progam to show figure (False by default)
    --------------------------------
    Returns None
    """
        #### Initializations ####
    plt.figure(figsize=(20,8))          
    plt.title(title,size=40,weight='bold')
    plt.xlabel("Frequency [Hz]",size=20,weight='bold')
    plt.ylabel("Amplitude",size=20,weight='bold')

    plt.plot(xdata,ydata,color='purple')
        
    plt.hlines(0,xdata[0],xdata[-1],color='black')
    plt.xlim(xdata[0],xdata[-1])
    plt.grid()
  
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.png')
    if show == True:
        plt.show()
    plt.close()