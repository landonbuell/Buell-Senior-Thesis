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
import scipy.sparse as sparse

import Math_Utilities as math_utils
import Plotting_Utilities as plot_utils

"""
Feature_Utilities.py - "Feature Extraction Utils"
    Contains Definitions to extract features and data from
    time-domain representations of a signal
"""

        #### CLASS DEFINITIONS ####

class Time_Series_Features ():
    """
    Extract feature information from signal data in time-domain
    --------------------------------
    waveform (arr) : Array of shape (1 x n_samples) representing 
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Time Series Feature Object
    """

    def __init__(self,waveform,npts=4096,overlap=0.75):
        """
        Instantiate Class Object 
        --------------------------------
        --------------------------------
        Return None
        """
        self.X = waveform                   # set waveform to self
        self.n_samples = self.X.shape[0]    # sample in waveform
        self.npts = npts                    # points per frame
        self.overlap = overlap              # overlap between frames
        self.frames = self.time_frames()    # create time-frames
        self.n_frames = self.frames.shape[0]# number of frames

    def time_frames (self):
        """
        Divide waveforms into frames of fixed length
            Waveform in tail-zero padded to adjust length
            Useful for building spectrogram
        --------------------------------
        * no args
        --------------------------------
        Return frames object (n_frames = 
        """
        frames = np.array([])               # array to hold time frames
        step = int(self.npts*(1-self.overlap))   # steps between frames
        for I in range(0,self.n_samples,step):      # iter through wave form
            x = X[I:I+npts]                 # create single frame
            if x.shape[0] != self.npts:          # frame w/o N samples
                pad = npts - x.shape[0]     # number of zeroes to pad
                x = np.append(x,np.zeros(shape=(1,pad)))  
            frames = np.append(frames,x)    # add single frame
        frames = frames.reshape(-1,npts)    # reshape (each row is frame)
        return frames                       # return frames

    def Phase_Space (self,dt=1):
        """
        Construct phase space representation of signal X
        --------------------------------
        dt (int) : sample spacing
        --------------------------------
        Return sparse matrix representation of phase-space
        """       
        dframes = np.gradient(self.frames,dt,axis=-1)  # 1st derivative
        phase_sparse_matrices = []                  # hold each sparse matrix
        for x,dx in zip(self.frames,self.dframes):  # in each frame...
            # Make sparse matrix
            phase = sparse.coo_matrix((np.ones(shape=n_samples),(x,dx)),
                                  shape=(n_samples,n_samples),dtype=np.int8)
        phase_sparse_matrices.append(phase)
        return phase

    def Rise_Decay_Time (self,low=0.1,high=0.9):
        """
        Extract rise/decay time parameter
        --------------------------------    
        low (float) : low band amplitude threshold
        hgih (float) : high bound amplitude threshold
        --------------------------------
        Return [risetime,decaytime] feature set
        """
        y = np.abs(self.X)                # abs of waveform
        above_low = np.where((y >= low) & (y <= high))[0]
        above_high = np.where((y >= high))[0]     
        rise = np.abs(above_high[0] - above_low[0])     # rise in samples
        decay = np.abs(above_high[-1] - above_low[-1])  # decay in samples
        return np.array([rise,decay])/self.rate         # scale by sample rate

    def RMS_above (self,RMS,vals=[(0,10),(10,25),(25,50),(50,75),(75,90)]):
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
        X_RMS = math_utils.RMS_Energy(self.frames)    # RMS of each frame
        counters = np.array([])             # arr to hold cntrs   
        for pair in vals:                   # each pair
            low,high = (pair[0]*RMS),(pair[1]*RMS)  # set bounds
            x = np.where((X_RMS>=low)&(X_RMS<high))[0]
            counters = np.append(counters,len(x))   # add number of frames
        counters /= self.n_frames           # normalize by number of frames
        return counters                     # return list of counter

class Frequency_Series_Features ():
    ""

