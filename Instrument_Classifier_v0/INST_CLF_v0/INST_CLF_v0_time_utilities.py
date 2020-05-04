"""
Landon Buell
Instrument Classifier v0
Time series Utility Functions
6 April 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd

import scipy.integrate as integrate

import INST_CLF_v0_base_utilities as base_utils

            #### FUNCTION DEFINITIONS ####

def rise_decay_time (waveform,start=0.1,stop=0.9):
    """
    Find portion of waveform to reach 'start' value to 'stop' value
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    start (float) : value (0,1) to indicate where to begin (0.1 by default)
    stop (float) : value (0,1) to indicate where to end (0.9 by default)
    --------------------------------
    Return rise  time parameter (0,1)
    """
    n_pts = len(waveform)                       # number of points in waveform
    abs_waveform = np.abs(waveform)             # compute abs val of waveform
    rise,decay = np.array([]),np.array([])      # imported indicies 
    for value in [start,stop]:                  # for start / stop amplitudes
        pts = np.where(abs_waveform >= value)[0]# pts above value
        rise = np.append(rise,pts[0])           # add 1st pt > start amp
        decay = np.append(decay,pts[-1])        # add last pt > stop amp
    rise_dt = np.abs(rise[1]-rise[0])           # pts to rise 10% - 90%
    decay_dt = np.abs(decay[1]-decay[0])        # pts to decay 90% - 10%
    # features are sensitive to scaling, so scale by 1/n_pts
    rise_frac = (rise_dt/n_pts)
    decay_frac = (decay_dt/n_pts)
    return rise_frac,decay_frac             # the return the two features

def Energy_Frames (waveform,n_samples=256,rate=44100):
    """
    Compute percentage of "frames" with RMS power less than
    given threshold.
    --------------------------------
    waveform (array) : 1 x N waveform from file with normalized amplitude
    n_samples (int) : Number of samples in single frame
    rate (int) : Audio sample rate in samples/sec
    --------------------------------
    return (int) number for frames with RMS power below 50% of 
    """
    ext_pts = len(waveform) % n_samples     # number of extra pts
    waveform = waveform[ext_pts:]           # truncate waveform
    waveform = waveform.reshape(-1,n_samples)   # reshape
    # Iterate through frames, collect energy for each
    frame_energies = np.array([])           # energy of each frame 
    for frame in waveform:                  
        frame = frame**2                    
        energy = integrate.trapz(y=frame,x=None,
                    dx=1/rate,axis=-1)      
        frame_energies = np.append(frame_energies,energy)
    # nroamlzie enrgies & compute RMS
    frame_energies /= np.max(frame_energies)
    RMS_energy = np.sqrt(np.sum(frame_energies**2)/len(frame_energies))
    return frame_energies,RMS_energy

def RMS_Below_Val (frame_energies,RMS,vals=[0.5]):
    """
    Find number of frames in waveform with RMS below given value
    --------------------------------
    frame_energies (arr) : 1 x M array containing normalize energies 
        of frame from waveforms
    RMS (float) : Root-Mean-Squared Energy Value of frames in waveform
    vals (iter) : (1 x k) arr containing fraction of RMS to beat as threshsold
    --------------------------------
    Return (1 x k) array, where i-th element is number of frames in waveform,
        with energy >= i-th val * RMS
    """
    n_frames = np.array([])         # output array
    for val in vals:                # each value to be greater than
        threshold = RMS*val         # threshold to beat:
        cntr = 0                    # n frame with more energy
        for frame in frame_energies:      
            if frame >= threshold:  # more energy
                cntr += 1           # increment counter
        n_frames = np.append(n_frames,cntr/len(frame_energies))
    # return number of frames w/ energy above each value
    return n_frames                 # 
                                    
def Time_Spectrum_Flux (X,dt=1):
    """
    Compute Spectral Flux in Time-Domain (Kahn,Wasfi,2006)
    --------------------------------
    X (arr) : 1 x M array containing time-series based data
    dt (float) : step in time (1 by default)
    --------------------------------
    Return array (1 x M-1) of TSF 
    """
    TSF = np.array([])          # time spectral flux output array
    for I in range (len(X)-1):  # in the X array
        dx = X[I+1] - X[I] 
        TSF = np.append(TSF,dx) # add to arr
    TSF /= dt                   # divide by int
    TSF /= np.max(np.abs(TSF))  # normalize
    return TSF                  # Return TSF

