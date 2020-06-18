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

import Math_Utilities as math_utils
import Freqspace_Features as freq_feats

            #### TIME SERIES FEATURES ####

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
    step = int(N*(1-overlap))       # steps between frames
    for I in range(0,FILE.n_samples,step):  # iter through wave form
        x = FILE.waveform[I:I+N]    # create single frame
        if x.shape[0] != N:         # frame w/o N samples
            pad = N - x.shape[0]    # number of zeroes to pad
            x = np.append(x,np.zeros(shape=(1,pad)))  
        frames = np.append(frames,x)    # add single frame
    frames = frames.reshape(-1,N)   # reshape (each row is frame)
    return frames                   # return frames

def Frames_fixed_number (FILE,N=10,overlap=0.75):
    """
    Divide waveforms into fixed number of frames
        Waveform is tail-zero padded to adjust length
        Useful for building amplitude envelope
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    N (int) : Fixed numer of samples to have in each frame
    overlap (float) : Percentage overlap between frames
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    pass

def rise_decay_time (FILE,low=0.1,high=0.9):
    """
    Extract rise/decay time parameter
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    y = np.abs(FILE.waveform)       # extract waveform
    above_low = np.where((y >= low) & (y <= high))[0]
    above_high = np.where((y >= high))[0]     
    rise = np.abs(above_high[0] - above_low[0])     # rise in samples
    decay = np.abs(above_high[-1] - above_low[-1])  # decay in samples
    return rise,decay                               # return rise & decay samples

def RMS_above (X,RMS,vals=[0.1,0.25,0.5,0.75,0.9]):
    """
    Compute percentage of frames with RMS energy greater than
        'val'% of total waveform RMS
    --------------------------------
    X (arr) : array-like of time-frame objects (n_frames x n_samples)
    RMS (float) : RMS energy wave for full waveform
    vals (iter) : percentages of RMS energy to beat (1 x n_vals)
    --------------------------------
    Return counter, array, shape (1 x n_vals)
    """
    X_RMS = math_utils.RMS_Energy(X)    # RMS of each frame
    counters = np.array([])             # arr to hold cntrs
    n_frames = X.shape[0]               # frames in X
    for val in vals:                    # for each value to beat
        cntr = 0                        # init counter
        threshold = RMS*val             # threshold to beat
        for x in X_RMS:                 # iter through frame RMS's
            if x > threshold:           # if greater than RMS
                cntr += 1               # inc cntr
        counters = np.append(counters,cntr)
    counters /= n_frames                # normalize by number of frames
    return counters                     # return list of counter



