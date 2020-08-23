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
    rate (int) : Sample rate of signal in Hz
    frames (arr) : Array of time frames, if None, a new array is created
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Time Series Feature Object
    """
    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,
                frames=frames,npts=npts,overlap=overlap)

    def __Call__(self):
        """
        Collect preset features from self in single function
        --------------------------------
        *no args
        --------------------------------
        Return features in time-domain
        """
        featureVector = np.array([])
        featureVector = np.append(featureVector,self.TimeDomainEnvelope())
        featureVector = np.append(featureVector,self.ZeroCrossingRate())
        featureVector = np.append(featureVector,self.CenterOfMass())
        featureVector = np.append(featureVector,self.WaveformDistributionData())
        featureVector = np.append(featureVector,self.AutoCorrelationCoefficients())
        return featureVector
    
    def TimeDomainEnvelope(self,attrb='signal'):
        """ 
        Compute Time-Envelope by waveform or by frame (Virtanen) 
        --------------------------------
        attrb (str) : Attribute to operate with. Must be in ['signal','frames']
        --------------------------------
        Return Time-Domain-Envelope of waveform or frames
        """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        TDE = np.sum(X*X,axis=-1)/self.n_samples
        TDE = np.sqrt(TDE)
        return TDE

    def ZeroCrossingRate (self,attrb='signal'):
        """ 
        Compute Zero-Crossing rate of signal (Kahn & Al-Khatib)
        --------------------------------
        attrb (str) : Attribute to operate with. Must be in ['signal','frames']
        --------------------------------
        Return zero-crossing rate of array
        """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        X = np.diff(np.sign(X),n=1,axis=-1)
        ZXR = np.sum(np.abs(X),axis=-1)
        return ZXR/2

    def CenterOfMass (self,attrb='signal'):
        """
        Compute temporal center of mass of waveform or each frame (Virtanen) 
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['signal','frames']
        --------------------------------
        Return temporal center of mass
        """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        weights = np.arange(0,X.shape[-1],1)    # weight array
        COM = np.matmul(X,weights)              # operate
        if COM.ndim > 1:                # more than 1D
            return np.mean(COM,axis=-1) # return average
        else:                           # scalar
            return COM/self.n_samples   # divide by n samples 

    def WaveformDistributionData (self,attrb='signal'):
        """ 
        Compute Distribution Data of Waveform Spectrum
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['signal','frames']
        --------------------------------
        return [mean,median,variance] of array or last axis of array
        """
        assert attrb in ['signal','frames']
        X = self.__getattribute__(attrb)    # isolate signal or frames
        return math_utils.MathematicalUtilities.DistributionData(X)

    def AutoCorrelationCoefficients (self,K=4):
        """ 
        Compute first K 'autocorrelation coefficients' from waveform (Virtanen) 
        --------------------------------
        K (int) : Number of coefficients to produce
        --------------------------------
        Retuen array of coefficients (1 x K)
        """
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
    rate (int) : Sample rate of signal in Hz
    frames (arr) : Array of time frames, if None, a new array is created
    npts (int) : Number of samples to include in each time-frame
    overlap (float) : Percentage overlap of sample between adjacent frames
    --------------------------------
    Return Instantiated Frequency Series Feature Object
    """

    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,
                frames=frames,npts=npts,overlap=overlap)

        # lambda function unit conversions
        self.HertzToMel = lambda h : 2595*np.log10(1+ h/700)
        self.MelToHertz = lambda m : 700*(10**(m/2595)-1)

        # Time Axis, Frequency Axis, Spectrogram
        self.hertz,self.frequencyPoints = self.FrequencyAxis(low=0,high=6000)
        self.mels = 2595*np.log10(1+self.hertz/700)
        self.t = np.arange(0,self.n_frames,1)   
        self.spectrogram = self.PowerSpectrum(pts=self.frequencyPoints).transpose()

    def __Call__(self):
        """
        Collect preset features from self in single function
        --------------------------------
        *no args
        --------------------------------
        Return features in frequency-domain
        """
        featureVector = np.array([])
        featureVector = np.append(featureVector,self.MelFrequencyCeptralCoefficients())
        featureVector = np.append(featureVector,self.CenterOfMass())
        return featureVector

    def FrequencyAxis (self,low=0,high=6000):
        """
        Compute Frequenxy Axis
        --------------------------------
        low (float) : Low value for frequency slice
        high (float) : High value for frequency bound
        --------------------------------
        Return frequency axis array between bounds, f
            and appropriate index, pts
        """
        self.lowHz,self.highHz = low,high                   # set low/high bnds in Hz
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
        return X                # return new window

    def PowerSpectrum (self,attrb='frames',pts=[]):
        """
        Compute Discrete Fourier Transform of arrays in X
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['signal','frames']
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

    def MelFilters (self,n_filters):
        """ 
        Compute the first 'm' Mel Frequency Ceptral Coefficients 
        --------------------------------
        n_filters (int) : Number of mel filters to use in frequency spectrum
        --------------------------------
        return filterBanks ( n_filters x self.npts) array
        """
        lowMelFreq = self.HertzToMel(self.lowHz)        # low bnd frequency
        highMelFreq = self.HertzToMel(self.highHz)      # high bnd frequency
        melPts = np.linspace(lowMelFreq,highMelFreq,n_filters+2)
        hertzPts = self.MelToHertz(melPts)              # convert to hz
        _bin = np.floor((self.npts+1)*hertzPts/self.rate)

        filterBanks = np.zeros((n_filters,self.npts),dtype=np.float32)
        for m in range (1,n_filters+1,1): # each filter
            freqLeft = int(_bin[m-1])
            freqCenter = int(_bin[m])
            freqRight = int(_bin[m+1])

            for k in range(freqLeft,freqCenter):
                filterBanks[m-1,k] = (k - _bin[m-1]) / (_bin[m] - _bin[m-1])
            for k in range(freqCenter,freqRight):
                filterBanks[m-1,k] = (_bin[m+1] - k) / (_bin[m+1] - _bin[m])
       
        filterBanks = filterBanks[:,:len(self.frequencyPoints)]
        return filterBanks

    def MelFrequencyCeptralCoefficients (self,attrb='spectrogram',n_filters=12):
        """ 
        Compute Mel Filter Bank Energies across full DFT or spectrogram 
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['frequencySeries','spectrogram']
        n_filters (int) : Number of mel filters to use in frequency spectrum (default = 12)
        --------------------------------
        Return MFCC applied to spectrum (self.n_frames/self.npts x n_filters)
        """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        X = X.transpose()
        melFiltersBanks = self.MelFilters(n_filters).transpose() # get mel filters
        MFCCs = np.matmul(X,melFiltersBanks)                    # apply to frequency spectrum
        if MFCCs.ndim > 1:                  # 2D array
            MFCCs = np.mean(MFCCs,axis=0)   # summ about 0-th axis
        return MFCCs

    def CenterOfMass (self,attrb='spectrogram'):
        """ 
        Compute frequency center of mass of spectrum or spectrogram (Virtanen) 
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['frequencySeries','spectrogram']
        --------------------------------
        return spectral center of mass
        """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        weights = np.arange(0,X.shape[-1],1)    # weight array
        COM = np.matmul(X,weights)              # operate
        if COM.ndim >= 1:                # more or equal to 1D
            return np.mean(COM,axis=-1) # return average
        else:                           # scalar
            return COM/self.n_samples   # divide by n samples 

    def FrequnecyDistributionData (self,attrb='spectrogram'):
        """ 
        Compute Distribution Data of Frequency Spectrum
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['frequencySeries','spectrogram']
        --------------------------------
        return [mean,median,variance] of array or last axis of array
        """
        assert attrb in ['frequencySeries','spectrogram']
        X = self.__getattribute__(attrb)    # isolate frequency or frames
        raise NotImplementedError
