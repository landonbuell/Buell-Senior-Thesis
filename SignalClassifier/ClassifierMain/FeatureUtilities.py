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

import MathUtilities as math_utils
import PlottingUtilities as plot_utils

"""
FeaturesUtilities.py - "Feature Extraction Utils"
    Contains Definitions to extract features and data from
    time-domain representations of a signal
"""

        #### CLASS DEFINITIONS ####

class BaseFeatures:
    """
    Basic Feature Extraction Class
        'TimeSeriesFeatures' and 'FrequencySeriesFeatures' inherit from here
    Assigns waveform attribute, sample rate, number of samples,
    --------------------------------
    waveform (arr) : 1 x N array of FP64 values representing a time-series waveform
    rate (int) : Signal sample rate in Hz (samples/sec)
    npts (int) : Number of samples used in each analysis frame
    overlap (float) : Percent overlap between frames (must be in (0,1))
    n_frames (int) : Number of desired analysis frames
    presetFrame (arr) : Array of analysis frames already made
    --------------------------------
    divides into time-frames
    """

    def __init__(self,waveform,rate=44100,npts=4096,overlap=0.75,n_frames=256,presetFrames=None):
        """ Initialize Class Object """
        self.signal = waveform              # set waveform to self
        self.n_samples = self.signal.shape[-1]  # samples in waveform
        self.rate = rate                    # sample rate
        self.npts = npts                    # points per frame
        self.overlap = overlap              # overlap between frames
        self.n_frames = n_frames            # number of desired frames     
        self.frameStep = int(self.npts*(1-self.overlap))  # steps between frames

    def ResizeWaveform(self):
        """ Truncate or Zero-Pad Waveform Depending on Length """
        currentSamples = self.signal.shape[-1]  # samples in waveform
        self.oldSamples = currentSamples        # number of sample before padding
        neededSamples = self.frameStep * (self.n_frames - 1) + self.npts
        if currentSamples > neededSamples:              # too many samples
            self.signal = self.signal[:neededSamples]   # take first needed samples
        elif currentSamples < neededSamples:            # not enough samples
            deficit = neededSamples - currentSamples    # sample deficit
            zeroPad = np.zeros((1,deficit),dtype=np.float64)
            self.signal = np.append(self.signal,zeroPad)# add the zero pad to array
        else:                                           # exactly right number of samples
            pass
        self.n_samples = self.signal.shape[-1]  # samples in waveform
        assert (self.n_samples == neededSamples)
        return self

    def AnalysisFrames (self):
        """
        Divide waveforms into analysis frames of fixed length
            Waveform in tail-zero padded to adjust length
            Useful for building spectrogram
        --------------------------------
        * no args
        --------------------------------
        Return frames object (n_frames x npts)
        """
        frames = np.array([])               # array to hold time frames
        step = self.frameStep               # iter step size
        framesInWaveform = int(np.floor(self.n_samples/step))    # frame from this waveform
        for i in range(0,framesInWaveform):                 # iter through wave form
            x = self.signal[(i*step):(i*step)+self.npts]    # create single frame 
            if len(x) < self.npts:                          # not enough samples
                deficit = self.npts - len(x)                # number of zeros to pad
                zeroPad = np.zeros((1,deficit),dtype=float) # create pad
                x = np.append(x,zeroPad)            # append pad to frame
            frames = np.append(frames,x)                    # add single frame
        frames = frames.reshape(framesInWaveform,self.npts)    # reshape (each row is frame)
        return frames                       # return frames

class TimeSeriesFeatures (BaseFeatures):
    """
    Extract feature information from signal data in time-domain
        Some methods can be applied to the full signal : attrb='signal'
        or applied to each time-frame : attrb='frames'
    --------------------------------
    waveform (arr) : 1 x N array of FP64 values representing a time-series waveform
    rate (int) : Signal sample rate in Hz (samples/sec)
    npts (int) : Number of samples used in each analysis frame
    overlap (float) : Percent overlap between frames (must be in (0,1))
    n_frames (int) : Number of desired analysis frames
    presetFrame (arr) : Array of analysis frames already made
    --------------------------------
    Return Instantiated Time Series Feature Object
    """
    def __init__(self,waveform,rate=44100,npts=4096,overlap=0.75,n_frames=256,presetFrames=None):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,npts=npts,overlap=overlap,
                         n_frames=n_frames,presetFrames=presetFrames)
        self.frames = None          # no analysis frames? (temp???)

    def __Call__(self):
        """
        Collect preset features from self in single function
        --------------------------------
        *no args
        --------------------------------
        Return features in time-domain
        """
        self.ResizeWaveform()           # resize waveform
        # Create feature vector array and add features
        featureVector = np.array([])
        featureVector = np.append(featureVector,self.TimeDomainEnvelope())
        featureVector = np.append(featureVector,self.ZeroCrossingRate())
        featureVector = np.append(featureVector,self.CenterOfMass())       
        featureVector = np.append(featureVector,self.AutoCorrelationCoefficients())
        # Crop Waveform if previously Shorter
        if self.oldSamples < self.n_samples:
            self.signal = self.signal[:self.oldSamples]

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
        weights = np.arange(0,X.shape[0],1)     # weight array
        COM = np.dot(weights,np.abs(X))         # operate
        if COM.ndim >= 1:                   # more or equal to 1D
            return np.mean(COM)             # return average
        else:                               # scalar
            return COM/self.n_samples       # divide by n samples 

    def AutoCorrelationCoefficients (self,K=4):
        """ 
        Compute first K 'autocorrelation coefficients' from waveform (Virtanen) 
        --------------------------------
        K (int) : Number of coefficients to produce (1-indexed)
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

class FrequencySeriesFeatures (BaseFeatures):
    """
    Extract feature information from signal data in time-domain
    --------------------------------
    waveform (arr) : 1 x N array of FP64 values representing a time-series waveform
    rate (int) : Signal sample rate in Hz (samples/sec)
    npts (int) : Number of samples used in each analysis frame
    overlap (float) : Percent overlap between frames (must be in (0,1))
    n_frames (int) : Number of desired analysis frames
    presetFrame (arr) : Array of analysis frames already made
    --------------------------------
    Return Instantiated Frequency Series Feature Object
    """

    def __init__(self,waveform,rate=44100,npts=4096,overlap=0.75,n_frames=256,presetFrames=None):
        """ Initialize Class Object Instance """
        super().__init__(waveform=waveform,rate=rate,npts=npts,overlap=overlap,
                         n_frames=n_frames,presetFrames=presetFrames)
        self.frames = self.AnalysisFrames()

        # lambda function unit conversions
        self.HertzToMel = lambda h : 2595*np.log10(1+ h/700)
        self.MelToHertz = lambda m : 700*(10**(m/2595)-1)

        # Time Axis, Frequency Axis
        self.SetFrequencyRange(0,12000)
        self.hertz,self.frequencyPoints = self.FrequencyAxis()
        self.mels = self.HertzToMel(self.hertz)
        self.t = np.arange(0,self.n_frames,1)   

    def SetFrequencyRange(self,low=0,high=6000):
        """ Set Frequency Axis Bounds """
        self.lowHz,self.highHz = low,high
        return self

    def __Call__(self):
        """
        Collect preset features from self in single function
        --------------------------------
        *no args
        --------------------------------
        Return features in frequency-domain
        """      
        # Create Spectrogram
        self.spectrogram = self.PowerSpectrum(pts=self.frequencyPoints).transpose()
        self.spectrogram = math_utils.MathematicalUtilities.PadZeros(self.spectrogram,self.n_frames)

        # Add Elements to Feature vector
        featureVector = np.array([])
        MFBEs = self.MelFilterBankEnergies(n_filters=16)
        MFCCs = self.MelFrequencyCeptralCoefficients(MFBEs)

        featureVector = np.append(featureVector,MFCCs)
        featureVector = np.append(featureVector,self.CenterOfMass())
        return featureVector

    def FrequencyAxis (self):
        """
        Compute Frequenxy Axis
        --------------------------------
        * no args
        --------------------------------
        Return frequency axis array between bounds, f
            and appropriate index, pts
        """
        f_space = fftpack.fftfreq(n=self.npts,d=1/self.rate)# comput freq space
        pts = np.where((f_space>=self.lowHz)&(f_space<=self.highHz))[0]   # get slices
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

    def MelFilterBankEnergies (self,attrb='spectrogram',n_filters=12):
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
        melFiltersBanks = self.MelFilters(n_filters).transpose()    # get mel filters
        MFBEs = np.matmul(X,melFiltersBanks)                        # apply to frequency spectrum
        if MFBEs.ndim > 1:                  # 2D array
            MFBEs = np.mean(MFBEs,axis=0)   # summ about 0-th axis
        return MFBEs

    def MelFrequencyCeptralCoefficients (self,melFilterEnergies):
        """ 
        Compute Mel Filter Bank Energies across full DFT or spectrogram 
        --------------------------------
        melFilterEnergies (arr) : Array (1 x N ) of MFBEs 
        --------------------------------
        Return MFCC applied to spectrum (self.n_frames/self.npts x n_filters)
        """
        n_filters = len(melFilterEnergies)
        m = np.arange(0,n_filters)
        MFCCs = np.zeros(shape=(n_filters))         # init MFCC array
        for i in range(n_filters):                  # each MFCC:          
            _log = np.log10(melFilterEnergies)
            _cos = np.cos((i+1)*(m+0.5)*np.pi/(n_filters))
            _coeff = np.dot(_log,_cos)              # compute dot product
            MFCCs[i] = _coeff
        return np.sqrt(2/n_filters)*MFCCs

    def CenterOfMass (self,attrb='spectrogram'):
        """ 
        Compute frequency center of mass of spectrum or spectrogram (Virtanen) 
        --------------------------------
        attrb (str) : Attribute to use for computations. Must be in ['frequencySeries','spectrogram']
        --------------------------------
        return spectral center of mass
        """
        assert attrb in ['frequencySeries','spectrogram']
        """ This feature has been changed - Update it in Main Classifier! """
        X = self.__getattribute__(attrb)        # isolate frequency or frames
        weights = np.arange(0,X.shape[0],1)     # weight array
        COM = np.dot(weights,np.abs(X))         # operate
        if COM.ndim >= 1:                # more or equal to 1D
            return np.mean(COM)         # return average
        else:                           # scalar
            return COM/self.n_samples   # divide by n samples 

