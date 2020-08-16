"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np

import scipy.fftpack as fftpack
import scipy.integrate as integrate
import scipy.signal as signal
import scipy.sparse as sparse
import scipy.stats as stats

"""
Feature_Utilities.py - "Feature Extraction Utils"
    Contains Definitions to extract features and data from
    time-domain representations of a signal
"""

        #### CLASS DEFINITIONS ####

class MathematicalUtilities :
    """
    Mathematical Utilites for feature processing
    --------------------------------
    * no args
    --------------------------------
    All methods are static
    """

    @staticmethod
    def DistributionData (X):
        """
        Analyze properties of an array of FP values
        --------------------------------
        X (arr) : (1 x N) Array of FP numbers to analyze as distribution
        --------------------------------
        Return array of [mean,median,mode,variance]
        """
        mean = np.mean(X,axis=-1)        # avg        
        median = np.median(X,axis=-1)    # median
        var = np.var(X,axis=-1)         # variance
        return np.array([mean,median,var])

    @staticmethod
    def ReimannSum (X,dx):
        """
        Compute Reimann Sum of 1D array X with sample spacing dx
        --------------------------------
        X (arr) : (1 x N) Array of FP numbers to compute Reimann Sum of
        dx (float) : Spacing between samples, 1 by default
        --------------------------------
        Return Reimann Sum approximation of array
        """
        return np.sum(X)*dx

class BaseFeatures:
    """
    Basic Feature Extraction Class
        'Time_Series_Features' and 'Frequency_Series_Features' inherit from here
    Assigns waveform attribute, sample rate, number of samples,
    divides into time-frames
    """

    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object """
        self.signal = waveform              # set waveform to self
        self.n_samples = self.signal.shape[0]   # samples in waveform
        self.rate = rate                    # sample rate
        self.npts = npts                    # points per frame
        self.overlap = overlap              # overlap between frames
        if frames is None:                  # if not givenframes
            self.frames = self.TimeFrames() # set create the frames
        else:                               # otherwise
            self.frames = frames            # set the framnes
        self.n_frames = self.frames.shape[0]# number of frames

    def TimeFrames (self):
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
        step = int(self.npts*(1-self.overlap))  # steps between frames
        for I in range(0,self.n_samples,step):  # iter through wave form
            x = self.signal[I:I+self.npts]           # create single frame
            if x.shape[0] != self.npts:         # frame w/o N samples
                pad = self.npts - x.shape[0]    # number of zeroes to pad
                x = np.append(x,np.zeros(shape=(1,pad)))  
            frames = np.append(frames,x)        # add single frame
        frames = frames.reshape(-1,self.npts)   # reshape (each row is frame)
        return frames                       # return frames

    

class TimeSeriesFeatures (BaseFeatures):
    """
    Extract feature information from signal data in time-domain
        Some methods can be applied to the full signal : attrb='signal'
        or applied to each time-frame : attrb='frames'
    --------------------------------
    waveform (arr) : Array of shape (1 x n_samples) representing 
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Time Series Feature Object
    """
    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,
                frames=frames,npts=npts,overlap=overlap)

    def TimeDomainEnvelope(self,attrb='signal'):
        """ Compute Time-Envelope by waveform or by frame (Virtanen) """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        TDE = np.sum(X*X,axis=-1)/self.n_samples
        TDE = np.sqrt(TDE)
        return TDE

    def ZeroCrossingRate (self,attrb='signal'):
        """ Compute Zero-Crossing rate of signal (Kahn & Al-Khatib) """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        X = np.diff(np.sign(X),n=1,axis=-1)
        ZXR = np.sum(np.abs(X),axis=-1)
        return ZXR/2

    def CenterOfMass (self,attrb='signal'):
        """ Compute temporal center of mass of waveform or each frame (Virtanen) """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        weights = np.arange(0,X.shape[-1],1) 
        COM = np.dot(X,weights)
        return COM/np.sum(X,axis=-1)

    def WaveformDistributionData (self,attrb='signal'):
        """ Compute Distribution data  """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        return MathematicalUtilities.DistributionData(X)

    def AutoCorrelationCoefficients (self,K=4):
        """ Compute first K 'autocorrelation coefficients' from waveform (Virtanen) """
        coefficients = np.array([])     # arr to hold k coeffs
        for k in range (1,K+1,1):       # for k coeffs      
            _a,_b = self.signal[0:self.n_samples-k],self.signal[k:]            
            sumA = np.dot(_a,_b)        # numerator
            sumB = np.dot(_a,_a)        
            sumC = np.dot(_b,_b)
            R = sumA / (np.sqrt(sumB)*np.sqrt(sumC))    # compute coefficient
            coefficients = np.append(coefficients,R)    # add to list of coeffs
        return coefficients             # return the coeffs

    def PhaseSpace (self,dt=1):
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

class FrequencySeriesFeatures (BaseFeatures):
    """
    Extract feature information from signal data in time-domain
    --------------------------------
    waveform (arr) : Array of shape (1 x n_samples) representing 
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Frequency Series Feature Object
    """

    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,
                frames=frames,npts=npts,overlap=overlap)

        # Time Axis, Frequency Axis, Spectrogram
        self.hertz,self.frequencyPoints = self.FrequencyAxis(low=0,high=6000)
        self.mels = 2595*np.log10(1+self.hertz/700)
        self.t = np.arange(0,self.n_frames,1)   
        self.spectrogram = self.PowerSpectrum(pts=self.frequencyPoints).transpose()

    def FrequencyAxis (self,low=0,high=6000):
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
        f_space = fftpack.fftfreq(n=self.npts,d=1/self.rate)# comput freq space
        pts = np.where((f_space>=low)&(f_space<=high))[0]   # get slices
        f_space = f_space[pts]                              # truncate space        
        return f_space,pts                                  # return space & pts

    def HanningWindow (self,X):
        """
        Apply Hanning window to each row in array X
        --------------------------------
        X (arr) Array-like of time-frames (n_frames x n_samples)
        --------------------------------
        Return X w/ Hann window applied to each row
        """
        window = signal.hanning(M=X.shape[-1],sym=False)  # window
        X = np.dot(X,window)
        return X            # return new window

    def PowerSpectrum (self,attrb='frames',pts=[]):
        """
        Compute Discrete Fourier Transform of arrays in X
        --------------------------------
        x (arr) : array to transform and compute power spectrum
            either shape (n_frames x n_samples) or (1 x n_samples)
        pts (iter) : int index values to keep in array 
        --------------------------------
        Return Z, array
        """        
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        Z = self.HanningWindow(X)   # apply Hanning Window
        Z = fftpack.fft(X,axis=-1)  # apply DFT
        Z = np.abs(Z)**2            # compute power:
        if Z.ndim > 1:              # more than 1D
            if pts is not None:     # selection of pts
                Z = Z[:,pts]        # slice
        else:                       # 1D arr
            if pts is not None:     # selection of pts
                Z = Z[pts]          # slice
        Z /= self.npts              # pts in FFT
        return Z                    # return Axis

    def CenterOfMass (self,attrb):
        """ Compute frequency center of mass of spectrum or spectrogram (Virtanen) """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        weights = np.arange(0,X.shape[-1],1)    
        COM = np.dot(X,weights)
        return COM/np.sum(X,axis=-1)

    def FrequnecyDistributionData (self,X=None):
        """ Compute Distribution data  """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)    # isolate frequency or frames
        return MathematicalUtilities.DistributionData(X)

    def MelFilterBanks (self,attrbs,n_filters=20):
        """ Compute Mel Filter Bank Energies across full DFT or spectrogram """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        melFreqLowBnd,melFreqHighBnd = self.mels[0],self.mels[-1]
        melPoints = np.linspace(melFreqLowBnd,melFreqHighBnd,n_filters)