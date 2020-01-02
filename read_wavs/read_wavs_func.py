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
    
    def pitch_to_freq (self,pitchdict):
        """ Assign numerical frequency based on pitch label """
        note = self.filename.split('.')[-3]     # isolate pitch string
        freq = pitchdict[str(note)]             # find frequency value
        setattr(self,'note',note)               # set attribute
        setattr(self,'freq',freq)               # set attribute
        return note,freq                        # return note name & frequency

    def read_raw_wav(self):
        """ Read Raw data from directory file """      
        rate,data = sciowav.read(self.filename) # read raw data
        data = np.transpose(data)               # tranpose
        L = data[0]/np.max(np.abs(data[0]))     # norm. L waveform
        R = data[1]/np.max(np.abs(data[1]))     # norm. R waveform
        setattr(self,'L_ch',L)                  # set attrb to self
        setattr(self,'R_ch',R)                  # set attrb to self
        setattr(self,'rate',rate)               # sample rate
        return L,R,rate                         # return values

    def timespace (self,rate,npts):
        """ Create timespace axis """
        time = np.arange(0,npts)    # create axis
        #time = time/rate            # divide by sample rate
        setattr(self,'time',time)   # set attribute
        return time                 # return the array

    def attack (self,value):
        """ Find ATTACK section of waveform """
        pass

    def decay (self,attr,value):
        """ Find DECAY section of waveform """
        pass

    def sustain (self,attr,value):
        """ Find SUSTAIN section of waveform """
        pass

    def release (self,attr,value):
        """ Find RELEASE section of waveform """
        pass

    def freqspace(self,npts=441000,rate=44100):
        """ Create frequency space axis """          
        fspace = fftpack.fftfreq(n=npts,d=rate)     # create f-axis
        pts = np.where((fspace>=0))                 # pos freqs
        fspace = fspace[pts]                        # index array
        setattr(self,'fspace',fspace)               # self self attrb
        return fspace                               # return f-axis

    def FFT (self,attr,npts=441000):
        """ Discrete Fast Fourier Transform """
        data = self.__getattribute__(attr)      # isolate attribute
        fftdata = fftpack.fft(x=data,n=npts)    # compute fft
        power = np.abs(fftdata)**2              # power spectrum
        power = power/np.max(power)             # normalize amplitude
        return power                            # return power & freq space

    def freq_band (self,attr,power,space,band):
        """ Break up FFT spectrum into single frequency band """
        pts = np.where((space>=band[1])&(space<=band[2]))
        space = space[pts]                      # isolate section of fspace
        power = power[pts]                      # corresponding power spect
        name = self.instrument+'-'+str(self.freq)+'-'+self.note+\
            '-'+str(attr)+'-'+str(band[0])+'band'   # name for plot & dataframe

        return space,power,name                 # return values

        #### FUNCTION DEFINITIONS ####

def to_CSV (name,data,labs):
    """ Write array of desired attributes to CSV file """
    data = np.array(data).round(8)
    data = np.transpose(data)
    frame = pd.DataFrame(data=data,columns=labs,dtype=float)
    frame.to_csv(name+'.txt',sep='\t')      # write CSV
    return frame                            # return dataframe

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
        
def output_paths (bands,parent='wav_data'):
    """ Create dictionaries of output paths for data to be stored to """
            #### Parent Directories ####
    wavepath = 'C:/Users/Landon/Documents/'+parent+'/waveforms'     # output for waveforms
    fftpath = 'C:/Users/Landon/Documents/'+parent+'/frequencies'    # output for FFT spectra
    spectpath = 'C:/Users/Landon/Documents/'+parent+'/spectrograms' # output for spectrograms
    data ={'Waveforms':wavepath,
            'Frequencies':fftpath,'Spectrograms':spectpath}         # dictionary to hold all paths
            #### Create Sub Directories ####
    for path in ['attack','decay','sustain','release','total']:      
        key,val = str(path),str(wavepath)+'/'+str(path)         # ket key:val pair
        data.update({key:val})                                  # add to dict
    for path in bands:                                          # in the frequnecy band tuple 
        key,val = str(path[0]),str(fftpath)+'/'+ \
            str(path[0])+'_'+str(path[1])+'-'+str(path[2])      # ket key:val pair
        data.update({key:val})                                  # add to dict
    return data                                                 # completed dictionary

def make_paths (paths_dict):
    """ Test if paths exist """
    for path in paths_dict.values():    # for each entry
        if os.path.exists(path):        # is the path exisits
            continue                    # do nothing
        else:                           # otherwise
            os.makedirs(path)           # make the path

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