"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import sys

import scipy.fftpack as fftpack
import scipy.integrate as integrate
import scipy.signal as signal
import scipy.sparse as sparse

import Timespace_Features as time_feats
import Plotting_Utilities as plot_utils

            #### TIME SERIES FEATURES ####

def Frequency_Axis (npts=4096,rate=44100,low=0,high=6000):
    """
    Compute Frequenxy Axis
    --------------------------------
    npts (int) : Number of samples in the axis
    rate (int) : waveform sample rate in Hz (44.1k by default)
    low (float) : Low value for frequency slice
    high (float) : High value for frequency bound
    --------------------------------
    Return frequency axis array between bounds, f
        and appropriate index, pts
    """
    f_space = fftpack.fftfreq(n=npts,d=1/rate)     # comput freq space
    pts = np.where((f_space>=low)&(f_space<=high))[0]   # get slices
    f_space = f_space[pts]                              # truncate space        
    return f_space,pts                                  # return space & pts

def Hanning_Window (X):
    """
    Apply hanning window to each row in array X
    --------------------------------
    X (arr) Array-like of time-frames (n_frames x n_samples)
    --------------------------------
    Return X w/ Hann window applied to each row
    """
    w = signal.hanning(M=X.shape[1],sym=False)  # window
    for x in X:         # each row in X
        x *= w          # apply window
    return X            # return new window
           
def Power_Spectrum (x,pts=[]):
    """
    Compute Discrete Fourier Transform of arrays in X
    --------------------------------
    x (arr) : array to transform and compute power spectrum
        either shape (n_frames x n_samples) or (1 x n_samples)
    pts (iter) : int index values to keep in array 
    --------------------------------
    Return Z, array
    """
    z = fftpack.fft(x,axis=-1)  
    Z = np.abs(z)**2    # compute power:
    if len(pts) != 0:   # selection of points:
        Z = Z[:,pts]   # subset of points
    return Z            # return DFT matrix

def Spectrogram (X,f,pts):
    """
    Compute spectrogram using time-frames of a signal
    --------------------------------
    X (arr) : Array-like containing times frames
    f (arr) : axis mapping to points in frequency space
    pts (iter) : int index values to keep in array
    --------------------------------
    Return spectrogram, frequency & time axes, Sxx,f,t
    """
    n_frames,n_samples = X.shape    # input shape
    X = Hanning_Window(X)           # apply Hann window
    Sxx = Power_Spectrum(X,pts)     # compute FFT of each row
    Sxx = Sxx.transpose()           # transpose
    t = np.arange(0,n_frames)       # time axis  
    Sxx = sparse.coo_matrix(Sxx)    # make into spmatrix
    return f,t,Sxx

def Energy_Spectral_Density (f,t,Sxx,rate=44100,bands=[(0,6000)]):
    """
    Compute Energy Spectral density Distribution 
    --------------------------------
    f (arr) : Axis to map pts to frequency space (1 x n_bins)
    t (arr) : Axis to map pts to time space (1 x n_frames)
    Sxx (arr) : 2D Spectrogram (n_bins x n_frames) time vs. frequency vs. amplitude
    rate (int) : waveform sample rate in Hz (44.1k by default)
    bands (arr) : Iterable containing bounds of frequency bands (n_pairs x 2)
    --------------------------------
    Reuturn
    """ 
    energy = np.array([])               # arr to hold energy
    Sxx = Sxx.toarray()                       # sparse into np arr
    for i,pair in enumerate(bands):     # each pair of bounds
        idxs = np.where((f>=pair[0])&(f<=pair[1]))[0]   # find f-axis idxs
        E = integrate.trapz(Sxx[idxs],dx=rate,axis=-1)  # integrate
        E = np.sum(E)                   # sum elements
        energy = np.append(energy,E)    # add to array
    energy /= len(t)                    # avg energy/frame
    energy /= np.max(energy)            # scale by max
    return energy                       # return



    

