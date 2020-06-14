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

import Freqspace_Features as freq_feats

            #### TIME SERIES FEATURES ####

def rise_decay_time (FILE,low=0.1,high=0.9):
    """
    Extract rise/decay time parameter
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    x = np.abs(FILE.waveform)       # extract waveform
    above_low = np.array([])        # above 10%
    above_high = np.array([])       # above 90%
    for i,y in enumerate(x):        # index, value in waveform
        if (y >= low) and (y <= high):
            above_low = np.append(above_low,i)      # add index
        elif (y >= high):
            above_high = np.append(above_high,i)    # add index
        else:
            continue
    rise = np.abs(above_high[0] - above_low[0])     # rise in samples
    decay = np.abs(above_high[-1] - above_low[-1])  # decay in samples
    return FILE.add_features([rise,decay])          # set & return 

def RMS_Energy (X):
    """
    compute RMS energy of object X.
        Output contains 1 less dimension
    --------------------------------
    X (arr) : Array-like of floats 
    --------------------------------
    Return RMS of array in X
    """
    if X.ndim == 1:                 # single dimesnion
        RMS = np.sqrt(np.means(X))  # compute RMS
    elif X.ndim > 1:                # more than 1 dim
        RMS = np.array([])          # hold RMS values
        for x in X:
            RMS = np.append(RMS,np.sqrt(np.means(x)))  
    return RMS

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
    for I in range(0,FILE.n_samples,step):  # iter through wave form
        try:                            # attempt to create frame
            x = FILE.waveform[I:I+N]    # create single frame
        except IndexError:              # index out of bounds  ?          
            x = FILE.waveform[I:-1]     # make frame
            pad = np.zeroes(shape=(1,N-len(x)))     # pad
            x = np.append(x,pad)        # add padding zeroes
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



