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
        self.instrument = file.split('.')[0]    # Instrument name
        self.

    def read_raw_wav(self):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw data
        data = np.transpose(data)               # tranpose
        L = data[0]/np.max(np.abs(data[0]))     # norm. L waveform
        R = data[1]/np.max(np.abs(data[1]))     # norm. R waveform
        setattr(self,'L_track',L)               # set attrb to self
        setattr(self,'R_track',R)               # set attrb to self
        setattr(self,'rate',rate)               # sample rate
        return L,R,rate                         # return values

    def timespace (self,rate,npts):
        """ Create timespace axis """
        time = np.arange(0,npts)    # create axis
        #time = time/rate            # divide by sample rate
        setattr(self,'time',time)   # set attribute
        return time                 # return the array

    def FFT (self):
        """ Discrete Fast Fourier Transform """
        pass

    def attack (self,value):
        """ Find attack section of waveform """

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

def output_paths ():
    """ Create series of Output Paths for data to be included in """
            #### Parent Directories ####
    wavepath = 'C:/Users/Landon/Documents/wav_data/waveforms'       # output for waveforms
    fftpath = 'C:/Users/Landon/Documents/wav_data/frequencies'      # output for FFT spectra
    spectpath = 'C:/Users/Landon/Documents/wav_data/spectrograms'   # output for spectrograms
    paths_dict = {'Waveforms':wavepath,
                'Frequencies':fftpath,'Spectrograms':spectpath}     # dictionary to hold all paths
            #### Create Sub Directories ####
    for path in ['attack','decay','sustain','release','full_L','full_R']:      
        key,val = str(path),str(wavepath)+'/'+str(path)     # ket key:val pair
        paths_dict.update({key:val})                        # add to dict
    for path in ['low','midlow','mid','midhigh','high','fft_L','fft_R']:      
        key,val = str(path),str(fftpath)+'/'+str(path)      # ket key:val pair
        paths_dict.update({key:val})                        # add to dict
    return paths_dict                                       # return the dictionary

def notefreq_dict ():
    """ Dictionary of Note Names to Frequency Values """
    notes = np.array([])                    # array to hold note names
    for octave in np.arange(0,9):           # iterate through octaves
        for letter in ['A','Bb','B','C','Db','D','Eb','E','F','Gb','G','Ab']:   
            name = letter+str(octave)       # create note name
            notes = np.append(notes,name)   # add to array
    steps = np.arange(-int(12*4),+int(12*5),1,dtype=int)
    freqs = np.array([440*(2**(n/12)) for n in steps]).round(2)
    notefreq = {}                           # Dictionary to note
    for I in range (len(steps)):            # each step
        notefreq.update({notes[I]:freqs[I]})# update the dictionary
    return notefreq                         # return the dictionary
        
    

def make_paths (paths_dict):
    """ Test if paths exist """
    for path in paths_dict.values():    # for each entry
        if os.path.exists(path):        # is the path exisits
            continue                    # do nothing
        else:                           # otherwise
            os.mkdir(path)              # make the path

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