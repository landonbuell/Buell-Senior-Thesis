"""
Landon Buell
PHYS 799
Read wav files - functions
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
        self.wavepath = root.replace('wav_audio','wav_waveforms')
        self.fftpath = root.replace('wav_audio','wav_Spectra')
        self.spectpath = root.replace('wav_audio','wav_Spectra')

    def make_paths (self):
        """ Test if paths exist """
        paths = [self.wavepath,self.fftpath,self.spectpath]
        for path in paths:          # for each path
            if os.path.exists(path) == False:
                os.mkdir(path)
            else:
                continue

    def read_raw_wav(self):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw data
        data = np.transpose(data)               # tranpose
        L = data[0]/np.max(data[0])             # norm. L waveform
        R = data[1]/np.max(data[1])             # norm. R waveform
        setattr(self,'L_track',L)               # set attrb to self
        setattr(self,'R_track',R)               # set attrb to self
        setattr(self,'rate',rate)               # sample rate
        return L,R,rate                         # return values

    def timespace (self,rate,npts):
        """ Create timespace axis """
        time = np.arange(0,npts)        # create axis
        #time = time/rate                # divide by sample rate
        setattr(self,'time',time)       # set attribute
        return time


    def freqspace(self,npts,rate):
        """ Create frequency space axis """          
        fspace = fftpack.fftfreq(n=npts,d=rate)     # create f-axis
        pts = np.where((fspace>=0)&(fspace<=4000))  # isolate 0 - 4000 Hz
        fspace = fspace[pts]                        # index array
        setattr(self,'fspace',fspace)               # self self attrb
        return fspace                               # return f-axis

    def array_to_CSV (self,attrs,fname):
        """ Write array of desired attributes to CSV file """
        output = {}                             # output dictionary
        for attr in attrs:                      # attrs to write
            data = self.__getattribute__(attr)  # isolate attribute
            output.update({attr:data})          # add arr to dictionary
        frame = pd.DataFrame(data=output,dtype=float)  
        frame.to_csv(name+'.txt',sep='\t',
                     float_format='%8.4f',mode='w') # write frame of csv
        return frame                                # return dataframe  


        #### FUNCTION DEFINITIONS ####

def read_directory(dir):
    """Read all files in given directory path"""
    file_objs = []                          # list to '.wav' hold files objs
    for roots,dirs,files in os.walk(dir):   # all objects in parent path
        for file in files:                  # files in list of files
            if file.endswith('.wav'):       # if '.wav' file
                wavs = wav_file(roots,file) # make instance
                file_objs.append(wavs)      # add to list 
    return file_objs                        # return the list of files

        #### PLOTTING & VISUALIZATION FUNCTIONS #####

def Plot_Time (obj,title,attrs=[],save=False,show=False):
    """
    Produce Matplotlib Figure of data in time domain
    --------------------------------
    obj (class) : object to plot attributes of
    title (str) : Title for figure to save as
    attrs (list) : list of attribute strings to plot data of
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

    for attr in attrs:
        try:
            data = obj.__getattribute__(attr)  # isolate attribute
            plt.plot(obj.time,data,label=str(attr))
        except:
            print("\n\tERROR! - Could not plot attribute:",attr)
        
    plt.hlines(0,obj.time[0],obj.time[-1],color='black')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.time.png')
    if show == True:
        plt.show()
    plt.close()

def Plot_Freq (obj,title,attrs=[],save=False,show=False):
    """
    Produce Matplotlib Figure of data in Frequency Domain
    Note: Must have executed FFT & Freq_Space methods before calling
    --------------------------------
    obj (class) : object to plot attributes of
    title (str) : Title for figure to save as
    attrs (list) : list of attribute strings to plot data of
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

    for attr in attrs:
        try:
            data = obj.__getattribute__(attr)  # isolate attribute
            plt.plot(obj.freq_space,data,label=str(attr))
        except:
            print("\n\tERROR! - Could not plot attribute:",attr)
        
    plt.hlines(0,obj.freq_space[0],obj.freq_space[-1],color='black')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    if save == True:
        plt.savefig(title+'.freq.png')
    if show == True:
        plt.show()
    plt.close()