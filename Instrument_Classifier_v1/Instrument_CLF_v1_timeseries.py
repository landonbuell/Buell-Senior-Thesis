"""
Landon Buell
Instrument Classifier v1
Time Series Functions 
3 February 2020
"""

            #### IMPORTS ####

import numpy as np
import os

import scipy.signal as signal

import Instrument_CLF_v1_func as func

"""
INSTRUMENT CLASSIFIER V1 - FREQUENCY SERIES FUNCTIONS   
    - Creating time axes
    - organizing waveform

"""


def time_axis (N,rate=44100):
    """
    create a time axis for tiem series data
    --------------------------------
    N (int) : number of points in data
    rate (int) : number of samples per second (44100 by default)
    --------------------------------
    return numpy array for time axis
    """
    return np.arange(N)/rate

def reshape_waveform (wavobj,M=(2**12)):
    """
    Produce time series features for 'data' attribute on wavobj instance
        Must have already run 'read_raw_wav()' method for this to work!
    --------------------------------
    wavobj (class) : Instance of particular wavfile object
    M (int) : number of features per file obj (recc. 2^N w/ N as int)
    --------------------------------
    Returns (N x M) array of features amd (M x 1) array of labels
    """
    ext = len(wavobj.data) % M  # remaining idx left over
    X = wavobj.data[ext:]       # crop waveform
    setattr(wavobj,'data',X)    # rest attribute
    return X                    # return matrix & targets

def attack_frac (wavobj,start=0.1,stop=0.9):
    """
    Compute attack time of waveform as fraction of full file length
        Amplitude of waveform must be normalize to max(waveform) == 1
    --------------------------------
    wavobj (inst) : Instance of .wav file exact feature from
    start (float) : statring ampltiude threshold to test, bounded by (0,1) 
    stop (float) : stopping amplitude threshold to test, bounded by (0,1)
    --------------------------------
    returns fraction of total file for rise time (i.e. bounded by (0,1])
    """
    waveform = np.abs(wavobj.data)  # extract waveform from instance
    n_pts = len(waveform)           # number of points in waveform
    start_idx = 0                   # set index counter 
    while waveform[start_idx] <= start:     # while less than my val
        start_idx += 1              # step through each index
    stop_idx = start_idx            # start from where we left off
    while waveform[stop_idx] <= stop:       # while less than each value 
        stop_idx += 1               # step through each index
    idx_diff = stop_idx - start_idx # index difference
    risetime = idx_diff / n_pts     # fraction of total file
    return risetime                 # return that value

def release_frac (wavobj,start=0.1,stop=0.9):
    """
    Compute decay time as fraction of full file length
        Amplitude of waveform must be normalize to max(waveform) == 1
    --------------------------------
    wavobj (inst) : Instance of .wav file exact feature from
    start (float) : statring ampltiude threshold to test, bounded by (0,1) 
    stop (float) : stopping amplitude threshold to test, bounded by (0,1)
    --------------------------------
    returns fraction of total file for rise time (i.e. bounded by (0,1])
    """
    waveform = np.abs(wavobj.data)  # extract waveform from instance
    n_pts = len(waveform)           # number of points in waveform
    start_idx = -1                  # set index counter 
    while waveform[start_idx] <= start:     # while less than my val
        start_idx -= 1              # step backwards through each index
    stop_idx = start_idx            # start from where we left off
    while waveform[stop_idx] <= stop:       # while less than each value 
        stop_idx -= 1               # step through each index
    idx_diff = np.abs(stop_idx - start_idx) # index difference
    risetime = idx_diff / n_pts     # fraction of total file
    return risetime                 # return that value

def max_amp (wavobj,ref=0.1):
    """
    Compute fraction of time to reach ref pt to max amp
        Amplitude of waveform must be normalize to max(waveform) == 1
    --------------------------------
    wavobj (inst) : Instance of .wav file exact feature from
    ref (float) : refrence ampltide to use as bounds
    --------------------------------
    """
    waveform = wavobj.data          # extract waveform from instance
    n_pts = len(waveform)           # number of points in waveform
    max_idx = np.argmax(waveform)   # index of max of waveform
    above_ref = np.where((waveform>=ref))           # find pts where waveform >= ref
    start_idx,stop_idx = above_ref[0],above_ref[-1] # isolate end pts
    start_to_max = np.abs(max_idx-start_idx)/n_pts  # % of time to reach max
    max_to_stop = np.abs(max_idx-stop_idx)/n_pts    # % of time to reach ref
    return start_to_max,max_to_stop                 # return values