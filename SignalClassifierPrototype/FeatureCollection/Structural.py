"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           Administrative.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys
import numpy as np

import scipy.io.wavfile as sciowav
import scipy.fftpack as fftpack

import Administrative
import CollectionMethods

        #### CLASS DEFINITIONS ####

class SampleIO:
    """ Sample IO Contains Data for Each Audio File to Read """

    def __init__(self,path,targetInt,targetStr=None):
        """ Constructor for SampleIO Instance """
        self._filePath      = path
        self._targetInt     = targetInt
        self._targetStr     = targetStr

    def __del__(self):
        """ Destructor for SampleIO Instance """
        pass

    # Getters and Setters

    def getFilePath(self) -> str:
        """ Return the File Path """
        return self._filePath

    def getTargetInt(self) -> int:
        """ Return Target Label as Int """
        return self._targetInt

    def getTargetStr(self) -> str:
        """ Return Target Label as Str """
        return self._targetStr

    def getPathExtension(self) -> str:
        """ Get File Type Ext """
        return self._filePath.split(".")[-1]

    # Public Interface

    def readSignal(self):
        """ Read The Audio From the indicate file """
        ext = self.getPathExtension()
        if (ext == "wav"):
            # Wav File
            return self.readFileWav()
        else:
            # Not Implements
            errMsg = "File extension: " + ext + " is not yet supported"
            raise RuntimeError(errMsg)


    # Private Interface

    def readFileWav(self):
        """ Read Data From .wav file """
        sampleRate,data = sciowav.read(self._filePath)
        data = data.astype(dtype=np.float32).flatten()
        waveform = data / np.max(np.abs(data))
        return SignalData(sampleRate,waveform)

    # Magic Methods
    
    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Sample: " + self.getPathExtension() + " " + self.getTargetStr()


class SignalData:
    """ Contain all signal Data """

    def __init__(self,sampleRate,samples=None):
        """ Constructor for SignalData Instance """
        self._sampleRate            = sampleRate
        self._waveform               = samples
        self._analysisFramesTime    = None
        self._analysisFramesFreq    = None
        self._melFreqCepstrumCoeffs = None
        self._autoCorrelationCoeffs = None
        self._zeroCrossingsPerFrame = None
        self._frameEnergyTime       = None
        self._frameEnergyFreq       = None

    def __del__(self):
        """ Destructor for SignalData Instance """
        self.clear()
       
    # Getters and Setters

    def getSampleRate(self):
        """ Get the Sample Rate """
        return self._sampleRate

    def setSampleRate(self,rate):
        """ Set the Sample Rate """
        self._sampleRate = rate
        return self

    def getSamples(self):
        """ Get the Signal Samples """
        return self._waveform

    def setSamples(self,data):
        """ Set the Signal Samples """
        self.clear()
        self._waveform = data
        return self

    def getSampleSpace(self):
        """ Get the Sample Spacing """
        return (1/self._sampleRate)

    def getNumAnalysisFramesTime(self):
        """ Get the Number of Time Series analysis frames """
        if (self.AnalysisFramesTime is None):
            return 0
        else:
            return self.AnalysisFramesTime.shape[0]

    def getNumAnalysisFramesFreq(self):
        """ Get the Number of Time Series analysis frames """
        if (self.AnalysisFramesFreq is None):
            return 0
        else:
            return self.AnalysisFramesFreq.shape[0]

    # Properties to Access Arrays

    @property
    def Waveform(self):
        """ Access Time-Series Samples """
        return self._waveform

    @property
    def AnalysisFramesTime(self):
        """ Access Time-Series Analysis Frames """
        return self._analysisFramesTime

    @property
    def AnalysisFramesFreq(self):
        """ Access Time-Series Analysis Frames """
        return self._analysisFramesFreq

    @property
    def MelFreqCepstrumCoeffs(self):
        """ Access Mel-Cepstrum Frequency Coefficients """
        return self._melFreqCepstrumCoeffs

    @property
    def AutoCorrelationCoeffs(self):
        """ Acces the Auto-Correlation Coefficients """
        self._autoCorrelationCoeffs

    @property
    def ZeroCrossingFrames(self):
        """ Acces the Zero Crossings Of Each Frame """
        return self._zeroCrossingsPerFrame

    @property
    def FrameEnergiesTime(self):
        """ Access Time-Series Frame Energies """
        return self._frameEnergyTime

    @property
    def FrameEnergiesFreq(self):
        """ Access Time-Series Frame Energies """
        return self._frameEnergyFreq

    # Public Interface

    def clear(self):
        """ Clear all Fields of the Instance """
        self._waveform               = None
        self._analysisFramesTime    = None
        self._analysisFramesFreq    = None
        self._melFreqCepstrumCoeffs = None
        self._autoCorrelationCoeffs = None
        self._zeroCrossingsPerFrame = None
        self._frameEnergyTime       = None
        self._frameEnergyFreq       = None
        return self

    def buildAnalysisFrames(self,frameParams=None):
        """ Build Time-Series AnalysisFrames """
        if (self.Waveform is None):
            # No Signal - Cannot Make Frames
            errMsg = "ERROR: So signal to make analysis Frames"
            raise RuntimeError(errMsg)

        # Create the Frames Constructor with Params

        return self

    # Private Interface

class AnalysisFramesParameters:
    """ AnalysisFramesParamaters contains 
    values to use when building Analysis Frames """

    def __init__(self,samplesPerFrame=1024,samplesOverlap=768,
                 headPad=1024,tailPad=2048,maxFrames=256):
        """ Constructor for AnalysisFramesParameters Instance """
        self._samplesPerFrame   = samplesPerFrame
        self._samplesOverlap    = samplesOverlap
        self._padHead           = headPad
        self._padTail           = tailPad
        self._maxFrames         = maxFrames
        self._framesInUse       = 0

    def __del__(self):
        """ Destructor for AnalysisFramesParameters Instance """
        pass

    # Getters and Setters

    def getSamplesPerFrame(self) -> int:
        """ Get the Number of Samples in Each Frame """
        return self._samplesPerFrame

    def getSamplesOverlap(self) -> int:
        """ Get the Number of Overlap Samples in Each Frame """
        return self._samplesOverlap

    def getSizeHeadPad(self) -> int:
        """ Get the Size of the Head Pad """
        return self._padHead

    def getSizeTailPad(self) -> int:
        """ Get the size of the Tail Pad """
        return self._padTail

    def getMaxNumFrames(self) -> int:
        """ Get the Max Number of Frames to Use """
        return self._maxFrames

    def getNumFramesInUse(self) -> int:
        """ Get the Number of Frames Currently in use """
        return self._framesInUse

    def getTotalFrameSize(self) -> int:
        """ Get total Size of Each Frame including padding """
        return self._padHead + self._samplesPerFrame + self._padTail

class AnalysisFramesConstructor:
    """ AnalysisFramesConstructor build time-series analysis frames 
    from an input signal """

    pass

class BatchData:
    """ Class To Hold Data for Each Batch of Samples """
        
    def __init__(self,batchIndex,numSamples,numFeatures):
        """ Constructor for BatchDataInstance """
        self._batchIndex    = batchIndex
        self._numSamples    = numSamples
        self._numFeatures   = numFeatures
        self._means         = np.zeros(shape=(numFeatures),dtype=float)
        self._variances     = np.zeros(shape=(numFeatures),dtype=float)
        
    def __del__(self):
        """ Destructor for BatchData Instance """
        pass

    # Getters and Setters

    def getBatchIndex(self) -> int:
        """ Get the Index of this Batch """
        return self._batchIndex

    def getNumSamples(self) -> int:
        """ Get the Number of Samples in the Batch """
        return self._numSamples

    def getNumFeatures(self) -> int:
        """ Get the Number of Features in the Batch """
        return self._numFeatures

    def getMeans(self):
        """ Get the Average of Each Feature """
        return self._means

    def getVariances(self):
        """ Get the Variance of each Feature """
        return self._variances

    # Public Interface


    # Private Interface
