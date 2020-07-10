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

class Base_Features:

    def __init__(self,waveform,npts=4096,overlap=0.75):
        """ Instantiate Class Object """
        self.X = waveform                       # set waveform to self
        self.n_samples = self.X.shape[0]        # sample in waveform
        self.npts = npts                        # points per frame
        self.overlap = overlap                  # overlap between frames
        self.frames = self.time_frames()        # create time-frames
        self.n_frames = self.frames.shape[0]    # number of frames

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
        for I in range(0,self.n_samples,step):  # iter through wave form
            x = X[I:I+npts]                 # create single frame
            if x.shape[0] != self.npts:          # frame w/o N samples
                pad = npts - x.shape[0]     # number of zeroes to pad
                x = np.append(x,np.zeros(shape=(1,pad)))  
            frames = np.append(frames,x)    # add single frame
        frames = frames.reshape(-1,npts)    # reshape (each row is frame)
        return frames                       # return frames


class Time_Series_Features (Base_Features):
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
        super()

    
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

class Frequency_Series_Features (Base_Features):
    """
    Extract feature information from signal data in time-domain
    --------------------------------
    waveform (arr) : Array of shape (1 x n_samples) representing 
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Frequency Series Feature Object
    """

    def __init__():
        """ Instantiated Class Object """
        pass

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

    def Mel_Filter_Banks (f,Sxx,n_filters,):
        """
        Compute Mel Filter Banks Spectral Energies
        --------------------------------
        f (arr) Array corresponding to frequency axis
        Sxx (arr) : 2D Spectrogram (n_bins x n_frames) time vs. frequency vs. amplitude
        n_filters (int) : Number of Mel filter banks in output
        --------------------------------
        Return Energy Approximation for each Mel Bank
        """
        mel = 1125 * np.log(1 + f/700)  # convert Hz to Mels

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
        ncols = x.shape[1]  # number of pts in DFT
        z = fftpack.fft(x,axis=-1)  
        Z = np.abs(z)**2    # compute power:
        if len(pts) != 0:   # selection of points:
            Z = Z[:,pts]    # subset of points
        return Z/ncols      # return DFT matrix

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
        Return array of sides (1 x n_pairs) for ESD in each pair
        """ 
        energy = np.array([])               # arr to hold energy
        try:                                # attempt
            Sxx = Sxx.toarray()             # sparse into np arr
        except:                             # if failure...
            pass                            # do nothing
        for i,pair in enumerate(bands):     # each pair of bounds
            idxs = np.where((f>=pair[0])&(f<=pair[1]))[0]   # find f-axis idxs
            E = integrate.trapz(Sxx[idxs],dx=rate,axis=-1)  # integrate
            E = np.sum(E)                   # sum elements
            energy = np.append(energy,E)    # add to array
        energy /= len(t)                    # avg energy/frame
        energy /= np.max(energy)            # scale by max
        return energy                       # return

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
        Sxx_shape = Sxx.shape           # original shape
        Sxx = np.array([0 if x<1 else x for x in Sxx.ravel()])
        Sxx = Sxx.reshape(Sxx_shape)    # Reshape
        Sxx = sparse.coo_matrix(Sxx,shape=Sxx_shape,dtype=np.float32)
        return f,t,Sxx

