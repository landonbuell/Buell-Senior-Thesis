"""
Landon Buell
Classifier Preprocessing Module
PHYS 799
16 August 2020
"""

        #### IMPORTS ####

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack

class FrequencySeriesFeatures :

    def __init__(self):
        """ Intialize Class Object Instance """
        self.rate = 44100
        self.npts = 4096
        self.hertz,self.pts = self.FrequencyAxis(0,8000)
        self.HertzToMel = lambda h : 2595*np.log10(1+ h/700)
        self.MelToHertz = lambda m : 700*(10**(m/2595)-1)
        self.mels = self.HertzToMel(self.hertz)

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
        self.lowHz,self.highHz = low,high                   # set low/high bnds in Hz
        f_space = fftpack.fftfreq(n=self.npts,d=1/self.rate)# comput freq space
        pts = np.where((f_space>=low)&(f_space<=high))[0]   # get slices
        f_space = f_space[pts]                              # truncate space        
        return f_space,pts                                  # return space & pts
        
    def MelFrequencyCeptsralCoefficients1 (self,n_filters=10):
        """ Compute the first 'k' Mel Frequency Ceptral Coefficients """
        lowMelFreq = self.HertzToMel(self.lowHz)
        highMelFreq = self.HertzToMel(self.highHz)
        melPts = np.linspace(lowMelFreq,highMelFreq,n_filters+2)
        hertzPts = self.MelToHertz(melPts)
        _bin = np.floor((self.npts+1)*hertzPts/self.rate)
        
        filterBank = np.zeros((n_filters,int(np.floor(self.npts/2+1))))
        for m in range(1,n_filters+1,1):
            fm_left = int(_bin[m-1]) 
            fm_center = int(_bin[m]) 
            fm_right = int(_bin[m+1])
            
            for k in range(fm_left,fm_center):
                filterBank[m-1,k] = (k - _bin[m-1]) / (_bin[m] - _bin[m-1])
            for k in range(fm_center,fm_right):
                filterBank[m-1,k] = (_bin[m+1] - k) / (_bin[m+1] - _bin[m])

        return filterBank

    def MelFrequencyCeptsralCoefficients (self,n_filters=10):
        """ Compute the first 'k' Mel Frequency Ceptral Coefficients """
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
       
        
        return filterBanks
        
        