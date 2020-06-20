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
import scipy.sparse

import Math_Utilities as math_utils
import Freqspace_Features as freq_feats
import Plotting_Utilities as plot_utils

            #### TIME SERIES FEATURES ####

def Frames_fixed_length (X,N=4096,overlap=0.75):
    """
    Divide waveforms into frames of fixed length
        Waveform in tail-zero padded to adjust length
        Useful for building spectrogram
    --------------------------------
    X (arr) : time-series waveform (1 x n_frames)
    N (int) : Fixed numer of samples to have in each frame
    overlap (float) : Percentage overlap between frames (0,1)
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    frames = np.array([])           # array to hold time frames
    step = int(N*(1-overlap))       # steps between frames
    for I in range(0,X.shape[0],step):  # iter through wave form
        x = X[I:I+N]    # create single frame
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

def Phase_Space (X,dt=1):
    """
    Construct phase space representation of signal X
    --------------------------------
    X (arr) : time-series waveform or time-frames (1 x n_frames)
    dt (int) : sample spacing
    --------------------------------
    Return sparse matrix representation of phase-space
    """
    dX = np.gradient(X,dt,axis=-1)
    return dX

def Rise_Decay_Time (X,rate=44100,low=0.1,high=0.9):
    """
    Extract rise/decay time parameter
    --------------------------------
    X (arr) : time-series waveform (1 x n_frames)
    rate (int) : waveform sample rate in Hz (44.1k by default)
    low (float) : low band amplitude threshold
    hgih (float) : high bound amplitude threshold
    --------------------------------
    Return [risetime,decaytime] feature set
    """
    y = np.abs(X)                # abs of waveform
    above_low = np.where((y >= low) & (y <= high))[0]
    above_high = np.where((y >= high))[0]     
    rise = np.abs(above_high[0] - above_low[0])     # rise in samples
    decay = np.abs(above_high[-1] - above_low[-1])  # decay in samples
    return np.array([rise,decay])/rate

def RMS_above (X,RMS,vals=[(0,10),(10,25),(25,50),(50,75),(75,90)]):
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
    for pair in vals:                   # each pair
        low,high = pair[0]*RMS,pair[1]*RMS      # set bounds
        x = np.where((X_RMS>=low)&(X_RMS<high))[0]
        counters = np.append(counters,len(x))   # add number of frames
    counters /= n_frames                # normalize by number of frames
    return counters                     # return list of counter



