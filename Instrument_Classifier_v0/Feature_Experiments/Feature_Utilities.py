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
        X (arr) : Array of FP number to analyze as distribution
        --------------------------------
        Return array of [mean,median,mode,variance]
        """
        X = X.ravel()           # flatten
        mean = np.mean(X)       # avg
        median = np.median(X)   # median
        mode,cnts = stats.mode(X,axis=None) # mode
        var = np.var(X)         # variance
        return np.array([mean,median,mode[0],var])

class BaseFeatures:
    """
    Basic Feature Extraction Class
        'Time_Series_Features' and 'Frequency_Series_Features' inherit from here
    Assigns waveform attribute, sample rate, number of samples,
    divides into time-frames
    """

    def __init__(self,waveform,rate=44100,frames=None,npts=4096,overlap=0.75):
        """ Initialize Class Object """
        self.X = waveform                   # set waveform to self
        self.n_samples = self.X.shape[0]    # samples in waveform
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
            x = self.X[I:I+self.npts]           # create single frame
            if x.shape[0] != self.npts:         # frame w/o N samples
                pad = self.npts - x.shape[0]    # number of zeroes to pad
                x = np.append(x,np.zeros(shape=(1,pad)))  
            frames = np.append(frames,x)    # add single frame
        frames = frames.reshape(-1,self.npts)    # reshape (each row is frame)
        return frames                       # return frames


class TimeSeriesFeatures (BaseFeatures):
    """
    Extract feature information from signal data in time-domain
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

    def TimeDomainEnvelope (self):
        """ Compute Time Domain envelope of waveform (Virtanen) """
        envelope = np.dot(self.X,self.X)
        envelope = np.sqrt(envelope/self.n_samples)
        return envelope

    def ZeroCrossingRate (self):
        """ Compute Zero-Crossing rate of signal (Kahn & Al-Khatib)"""
        zeroXing = np.sign(np.diff(self.X))
        zeroXing = np.sum(zeroXing)
        return zeroXing

    def CenterOfMass (self):
        """ Compute temporal center of mass of waveform (Virtanen)"""
        weights = np.arange(0,self.n_samples,1) 
        centerOfMass = np.dot(weights,self.X)
        return centerOfMass/np.sum(self.X)

    def WaveformDistribution(self):
        """ Compute Distribution data from waveform """
        return MathematicalUtilities.DistributionData(self.X)

    def AutoCorrelationCoefficients (self,K=4):
        """ Compute first K 'autocorrelation coefficients' from waveform (virtanen) """
        coefficients = np.array([])     # arr to hold k coeffs
        for k in range (1,K+1,1):       # for k coeffs      
            _a,_b = self.X[0:self.n_samples-k],self.X[k:]            
            sumA = np.dot(_a,_b)          # numerator
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
        self.f,self.f_pts = self.FrequencyAxis(low=0,high=6000)
        self.t = np.arange(0,self.n_frames,1)          
        self.Sxx = self.PowerSpectrum(self.f_pts)

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

    def MelFilterBanks (f,Sxx,n_filters,):
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

    def PowerSpectrum (self,pts=[]):
        """
        Compute Discrete Fourier Transform of arrays in X
        --------------------------------
        x (arr) : array to transform and compute power spectrum
            either shape (n_frames x n_samples) or (1 x n_samples)
        pts (iter) : int index values to keep in array 
        --------------------------------
        Return Z, array
        """        
        Z = fftpack.fft(self.frames,axis=-1)  
        Z = np.abs(Z)**2        # compute power:
        if pts is not None:     # selection of pts
            Z = Z[:,pts]        # slice
        Z = np.transpose(Z)/self.npts   # pts in FFT
        return Z

    def EnergySpectralDensity (self,f,t,Sxx,rate=44100,bands=[(0,6000)]):
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

    def Spectrogram (self,X,f,pts):
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
