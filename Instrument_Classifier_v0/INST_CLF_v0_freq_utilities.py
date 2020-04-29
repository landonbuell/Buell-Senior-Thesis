"""
Landon Buell
Instrument Classifier v0
Frequency series Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import matlab

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
    rate (int) : Audio sample rate in samples/sec
    --------------------------------
    Return frequnecy space axis [0,6000] Hz
    """
    resolution = rate/n_pts                                 # FFT resolution
    frequency_space = fftpack.fftfreq(n=n_pts,d=1/rate)     # create f space
    pts = np.where((frequency_space>=0)&(frequency_space<=6000))
    frequency_space = frequency_space[pts]                  # truncate 0 - 6kHz    
    return frequency_space,pts,resolution                   # axis,idx,res

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

def Spectrogram (waveform,pts,N=2**10):
    """
    Compute spectrogram of audio file data
        (N x M) Frequency vs. Time matrix
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    pts (list) : list of pts to keep in FFT spectrum (num of rows in Spectrogram)
    N (int) : Number of points to use in each FFT (recc. 2^p where p is int)
    --------------------------------
    Return spectrogram of signal
    """
    step = int(N/4)             # step between frames (overlap)
    Sxx = np.array([])          # init spectrogram
    #pts = np.arange(0,N,1)

    for I in range (0,len(waveform)-N,step):    # iter throught waveform
        frame = waveform[I:I+N]                 # audio segment - 'frame'
        frame = Hanning_Window(frame)           # apply Hann window
        pwr = Power_Spectrum(frame,pts)         # pwr spectrum for sample
        Sxx = np.append(Sxx,pwr)                # add pwr spectrum

    Sxx = Sxx.reshape(len(pts[0]),-1)   # reshape
    f = np.arange(0,Sxx.shape[0])
    t = np.arange(0,Sxx.shape[1])
    return t,f,Sxx



def CSPE_MATLAB(waveform,n_pts):
    """
    Compute "Complex-Spectral-Phase-Evolution" of signal 
        See 'CSPE.m matlab' script for more details
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    n_pts (int) : number of point to include in CSPE window
    --------------------------------

    """
    MATLAB_ENG = matlab.engine.start_matlab()   # start MATLAB engine
    Xout,Yout = MATLAB_ENG.CSPE(indat=waveform,
                    varargin=['windowed'])
    return None
    

def Find_Peaks (spectrum,hgt,res):
    """
    Find Number of peaks above certain value in a given spectrum
    --------------------------------
    spectrum (arr) : 1 x N power spectrum to find peaks in 
    hgt (float) : minimum tolerance of height of spike
    res (float) : frequncy resolution in Hz
    --------------------------------
    Return number of peaks in spectrum
    """
    peaks = np.where((spectrum >= height)&(spectrum <= 1))[0]
    print(len(peaks),'\n')