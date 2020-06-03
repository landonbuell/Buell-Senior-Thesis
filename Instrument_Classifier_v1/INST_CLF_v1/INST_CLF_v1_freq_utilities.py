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

import INST_CLF_v1_base_utilities as base_utils

import scipy.fftpack as fftpack
import scipy.signal as signal
import scipy.integrate as integrate

"""
INSTRUMENT CLASSIFER v1 - FREQUENCY UTILITIES
        Functions related to producing and returning features for 
        design matrix in frequency domain
    - Hanning_Window
    - Frequency_Space
    - Power_Spectrum
    - Spectrogram
    - CSPE_MATLAB
    - Frequency_Banks
"""

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

def Power_Spectrum (waveform,norm=False,pts=[]):
    """
    Compute power spectrum of waveform using frequnecy space
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    norm (Bool)
    pts (list) : list of pts to keep in FFT spectrum
    
    --------------------------------
    Return power spectrum of shape (1  x len(pts)) size
    """
    fftdata = fftpack.fft(waveform,n=len(waveform),axis=-1)
    power = np.abs(fftdata)**2      # compute power spect
    if norm == True:                # if normalize req.
        power /= np.max(power)      # normalize
    return power[pts]               # return specific idx of pwr

def Spectrogram (waveform,rate=44100,N=2**10,overlap=0.75,
                 start_frame=0,end_frame=-1):
    """
    Compute spectrogram of audio file data
        (N x M) Frequency vs. Time matrix
    --------------------------------
    waveform (array) : 1 x M waveform from file with normalized amplitude
    rate(int) : Inverse of sampling frequency
    N (int) : Number of samples per frame used to compute FFT (recc 2^p)
    overlap (float) : percentage of overlap between adjacent frames (0,1)
    --------------------------------
    Return spectrogram of signal
    """
    # Initialize 
    step = int(N*(1 - overlap)) # step between frames 
    Sxx = np.array([])          # init spectrogram
    cntr = 0                    # number of FFT's computed

    # Compute frequency axis
    f,f_pts,f_resol = Frequency_Space(N,rate=rate)

    # Build Spectrogram
    for I in range (0,len(waveform)-N,step):    # iter throught waveform
        frame = waveform[I:I+N]                 # audio segment - 'frame'
        frame = Hanning_Window(frame)           # apply Hann window
        pwr = Power_Spectrum(frame,False,f_pts) # pwr spectrum for sample
        Sxx = np.append(Sxx,pwr)                # add pwr spectrum to matrix
        cntr +=1                                # increment counter
    Sxx = Sxx.reshape(cntr,-1).transpose()      # reshape
    
    # Compute time-axis
    t = np.arange(0,cntr,1)
    
    # Return time-axis, freq-axis and Sxx matrix
    return f,t,Sxx

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
    
def Frequency_Banks(f,t,Sxx,bnd_pairs=[(0,6000)]):
    """
    Compute power within a set of bands of the frequency spectrum
    --------------------------------
    f (arr) : (1 x N) frequency space axis
    t (arr) : (1 x M) time space axis
    Sxx (arr) : (N x M) matrix representing file's spectrogram
    bnd_pairs (iter) : iterable of bounds of frequnecy bands
        recc shape: [(a,b),(b,c),(c,d),...,(x,y),(y,z)]
    --------------------------------
    Return arr of powers in each frequency band
    """
    n_banks = len(bnd_pairs)        # number of banks to use
    n_frames = t.shape[0]           # number of frames in file
    Sxx = Sxx.transpose()               # transp spectrogram
    bank_pwrs = np.zeros((1,n_banks))   # arr to hold power/banks

    # Build Frequency Bank Array
    for frame in Sxx:                       # iterate through time
        frame_pwrs = np.array([])           # bank pwrs for frame

        for pair in bnd_pairs:                          # each pair of bnds
            low_bnd,high_bnd = pair[0],pair[1]          # establish bnds
            pts = np.where((f>=low_bnd)&(f<=high_bnd))  # find idxs of bands
            bank = frame[pts]                           # isolate band
            pwr = integrate.trapz(bank)                 # compute def integral
            frame_pwrs = np.append(frame_pwrs,pwr)      # add to arr
    
        bank_pwrs += frame_pwrs             # add frame power to bank

    bank_pwrs /= n_frames           # average over number of frames
    return bank_pwrs.ravel()        # return bank powers as arr of floats        