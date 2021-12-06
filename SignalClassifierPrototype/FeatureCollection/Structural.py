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
        return SignalData(sampleRate,data)

    # Magic Methods
    
    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Sample: " + self.getPathExtension() + " " + self.getTargetStr()


class SignalData:
    """ Contain all signal Data """

    def __init__(self,sampleRate,samples=None):
        """ Constructor for SignalData Instance """
        self._sampleRate            = sampleRate
        self._samples               = samples
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
        return self._samples

    def setSamples(self,data):
        """ Set the Signal Samples """
        self.clear()
        self._samples = data
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
    def Samples(self):
        """ Access Time-Series Samples """
        return self._samples

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
        self._samples               = None
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
        if (self.Samples is None):
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

class FeatureVector:
    """ Class to Hold Feature Data for a single Sample """

    def __init__(self,sampleShape):
        """ Constructor for FeatureVector Instance """
        self._sampleShape   = sampleShape
        self._data          = np.zeros(shape=sampleShape,dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureVector Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self):
        """ Get the Shape of this Sample """
        return self._sampleShape

    # Public Interface

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self._sampleShape,dtype=np.float32)
        return self

    # Magic Method

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __getitem___(self,key):
        """ Get the Item at the Index """
        return self._data[key]

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        value = np.float32(value)   # cast to single-precs
        self._data[key] = value
        return self

class DesignMatrix:
    """ Class To hold Design Matrix """

    def __init__(self,numSamples: int,sampleShape: tuple):
        """ Constructor for DesignMatrix Instance """
        self._numSamples    = numSamples 
        self._sampleShape   = sampleShape
        self._data          = np.zeros(shape=self.getShape(),dtype=np.float32)

    def __del__(self):
        """ Destructor for DesignMatrix Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self) -> int:
        """ Get Total Shape of Design Matrix """
        return ((self._numSamples,) + self._sampleShape)

    def getNumSamples(self) -> int:
        """ Get the Number of Samples in the Design Matrix """
        return self._numSamples

    def setNumSamples(self,numSamples):
        """ Set the Number of Samples in the Design Matrix """
        self._numSamples = numSamples
        self.clearData()
        return self

    # public Interface

    def serialize(self,path=None):
        """ Write this design matrix out to a file """
        return self

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self.getShape(),dtype=np.float32)
        return self

    # Private Interface



    # Magic Methods 

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __getitem___(self,key):
        """ Get the Item at the Index """
        if (key < 0 or key >= self._numSamples):
            errMsg = "key index is out of range for " + self.__repr__
            raise IndexError(errMsg)

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        value = np.float32(value)   # cast to single-precs
        self._data[key] = value
        return self

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

    def getMeans(self) -> int:
        """ Get the Average of Each Feature """
        return self._mean

    def setMeans(self,values):
        """ Set the mean of Each Feature """
        if (values.shape[0] != self._numFeatures):
            errMsg = "Incorrect shape for number of features"

