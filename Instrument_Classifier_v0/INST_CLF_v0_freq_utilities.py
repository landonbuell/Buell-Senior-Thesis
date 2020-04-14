"""
Landon Buell
Instrument Classifier v0
Frequency series Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd

import scipy.fftpack as fftpack
import scipy.signal as signal

            #### FUNCTION DEFINITIONS ####
           
def Hanning_Window (waveform):
    """
    Apply Hanning Window Taper to waveform
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    --------------------------------
    Return waveform with hanning window applied
    """
    return waveform*signal.hanning(M=len(waveform))

def Frequency_Space (n_pts,rate=44100):
    """
    Build frequency space axis for waveform
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    --------------------------------
    Return frequnecy space axis [0,6000] Hz
    """
    frequency_space = fftpack.fftfreq(n=n_pts,d=1/rate)     # create f space
    pts = np.where((frequency_space>=0)&(frequency_space<=6000))
    frequency_space = frequency_space[pts]                  # truncate 0 - 6kHz
    return frequency_space , pts                            # return axis & idxs

def Power_Spectrum (waveform,pts):
    """
    Compute power spectrum of waveform using frequnecy space
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    pts (list) : list of pts to keep in FFT spectrum
    --------------------------------
    Return power spectrum of shape (1  x len(pts)) size
    """
    fftdata = fftpack.fft(waveform,n=len(waveform),axis=-1)
    power = np.abs(fftdata)**2                  # compute power spect
    power /= np.max(power)                      # normalize
    return power[pts]                           # return specific idx of pwr

def Find_Peaks (spectrum,height):
    """
    Find Number of peaks above certain value in a given spectrum
    --------------------------------
    spectrum (arr) : 1 x N power spectrum to find peaks in 
    height (float) : minimum tolerance of height of spike
    --------------------------------
    Return number of peaks in spectrum
    """
    peaks = np.where((spectrum >= height)&(spectrum <= 1))[0]
    print(len(peaks),'\n')