"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as sciowav


"""
SystemUtilities.py - "SystemUtilities"
    Contains Variables, Classes, and Definitions for Lower program functions
    Backends, Data structure objects, os & directory organization and validations
"""

class ProgramInitializer:
    """
    Object to handle all program preprocessing
    --------------------------------
    * no args
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self):
        """ Initialize Class Object Instance """
        self.wavFiles = self.CollectFiles()         # Get all .wav files in CWD
        self.n_files = len(self.wavFiles)           # get number of files   

    def CollectFiles (self,exts='.wav'):
        """ Search Local Directory for all files matching given extension """
        extFiles = []
        for file in os.listdir(os.getcwd()):    # in the path
            if file.endswith(exts):             # is proper file type
                extFiles.append(file)           # add to list
        return extFiles                         # return the list

def ReadFileWAV (filename):
    """ Read raw .wav file data from local path """
    rate,data = sciowav.read(filename)      # read .wav file
    data = data.reshape(1,-1).ravel()       # flatten waveform
    waveform = data/np.abs(np.amax(data))   # norm by max amp
    return waveform                         # return waveform

def PlotSpectrogram (f,t,Sxx,nameSave='',
                      save=False,show=True):
    """
    Create visualization of soundwave as frequency vs. time vs. power
    --------------------------------
    f (arr) : (1 x N) frequency space axis
    t (arr) : (1 x M) time space axis
    Sxx ((arr) : (N x M) matrix representing file's spectrogram
    title (str) : Title for plot
    save (bool) : If true, save MPL figure to cwd (False by Default)
    show (bool) : If true, shows current figure to User (True by Default)
    --------------------------------
    return None
    """
    plt.figure(figsize=(16,12))
    plt.xlabel('Time [Frame Index]',size=30,weight='bold')
    plt.ylabel('Frequnecy [Hz]',size=30,weight='bold')

    plt.pcolormesh(t,f,Sxx,cmap=plt.cm.plasma)

    plt.grid()
    plt.tight_layout()
    if save == True:
        plt.savefig(nameSave+'.png')
    if show == True:
        plt.show()