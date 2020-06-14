"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import scipy.fftpack as fftpack
import scipy.integrate as integ
import scipy.signal as signal

            #### TIME SERIES FEATURES ####

def rise_decay_time (FILE,low=0.1,high=0.9):
    """
    Extract rise/decay time parameter
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    pass

def RMS_Energy (FILE):
    """
    compute RMS energy of waveform
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    pass

def Frames_fixed_length (FILE,N=4096,overlap=0.75):
    """
    Divide waveforms into frames of fixed length
        Waveform in tail-zero padded to adjust length
        Useful for building spectrogram
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    N (int) : Fixed numer of samples to have in each frame
    overlap (float) : Percentage overlap between frames (0,1)
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    frames = np.array([])           # array to hold time frames
    step = N*(1-overlap)            # steps between frames
    for I in range(0,(FILE.n_samples-N),step):  # iter through wave form
        x = FILE.waveform[I:I+N]                 # create single frame
        frames = np.append(frames,x)    # add single frame
    frames = frames.reshape(-1,N)   # reshape (each row is frame)
    return frames                   # return frames

def Frames_fixed_number (FILE,N=10,overlap=0.75):
    """
    Divide waveforms into fixed number of frames
        Waveform in tail-zero padded to adjust length
        Useful for building amplitude envelope
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    N (int) : Fixed numer of samples to have in each frame
    overlap (float) : Percentage overlap between frames
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    pass


            #### FREQUENCY SERIES FEATURES ####


def Assemble_Features (FILE):
    """
    Create & Collect all classification features
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return (1 x N) array of features
    """

    
    return FILE                    # return file with feature vector


