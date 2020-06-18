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

import Timespace_Features as time_feats

            #### TIME SERIES FEATURES ####

def Frequency_Axis (FILE,npts=4096,low=0,high=6000):
    """
    Compute Frequenxy Axis
    --------------------------------
    FILE (inst) : file_object instance with file.waveform attribute
    npts (int) : Number of samples in the axis
    low (float) : Low value for frequency slice
    high (float) : High value for frequency bound
    --------------------------------
    Return frequency axis array between bounds, f
        and appropriate index, pts
    """
    freq = fftpack.fftfreq(n=npts,d=1/FILE.rate)    # comput freq space
    pts = np.where((f_space>=low)&(f_space<=high))  # get slices
    freq = freq[pts]                                # truncate space        
    return freq,pts                                 # return space & pts
            
def Power_Spectrum (FILE,pts=None):
    """
    Compute Discrete Fourier Transform of arrays in X
    --------------------------------
    X (arr) : array to transform and compute power spectrum
        either shape (n_frames x n_samples) or (1 x n_samples)
    pts (iter) : int index values to keep in array 
    --------------------------------
    Return Z, array
    """
    z = fftpack.fft(FILE.waveform,axis=-1)  
    Z = np.abs(z)**2    # compute power:
    if pts != None:     # selection of points:
        Z = Z[:][pts]   # subset of points
    return Z            # return DFT matrix

