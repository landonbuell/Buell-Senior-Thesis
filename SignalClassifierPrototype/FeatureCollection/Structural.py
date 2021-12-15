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
import matplotlib.pyplot as plt

import scipy.io.wavfile as sciowav
import scipy.fftpack as fftpack
import scipy.signal as scisig

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
        self._reqSamples    = int(2**18)

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
        waveform = self.padWaveform(waveform)
        return SignalData(sampleRate,waveform)

    def padWaveform(self,waveform):
        """ Pad or Crop Waveform if too long or too short """
        if (waveform.shape[0] < self._reqSamples):
            # Too few samples
            deficit = self._reqSample - waveform.shape[0]
            waveform = np.append(waveform,np.zeros(shape=deficit,dtype=np.float32))
        elif (waveform.shape[0] > self._reqSamples):
            # Too many samples
            waveform = waveform[0:self._reqSamples]
        else:
            # Exactly the right number of samples - do nothing
            pass
        return waveform

    # Magic Methods
    
    def __repr__(self) -> str:
        """ Debug representation of instance """
        return "Sample: " + self.getPathExtension() + " " + self.getTargetStr()


class SignalData:
    """ Contain all signal Data (NOT ENCAPSULATED) """

    def __init__(self,sampleRate,samples=None):
        """ Constructor for SignalData Instance """
        self._sampleRate            = sampleRate
        self.Waveform               = samples
        self.AnalysisFramesTime     = None
        self.AnalysisFramesFreq     = None
        self.MelFreqCepstrumCoeffs  = None
        self.AutoCorrelationCoeffs  = None
        self.ZeroCrossingsPerFrame  = None
        self.FrameEnergyTime        = None
        self.FrameEnergyFreq        = None

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

    # Public Interface

    def clear(self):
        """ Clear all Fields of the Instance """
        self._waveform              = None
        self._analysisFramesTime    = None
        self._analysisFramesFreq    = None
        self._melFreqCepstrumCoeffs = None
        self._autoCorrelationCoeffs = None
        self._zeroCrossingsPerFrame = None
        self._frameEnergyTime       = None
        self._frameEnergyFreq       = None
        return self

    def makeAnalysisFramesTime(self,frameParams=None):
        """ Build Time-Series AnalysisFrames """
        if (self.Waveform is None):
            # No Signal - Cannot Make Frames
            errMsg = "ERROR: So signal to make analysis Frames"
            raise RuntimeError(errMsg)

        # Create the Frames Constructor with Params
        constructor = AnalysisFramesConstructor(self,frameParams)
        constructor.call()

        return self

    # Private Interface

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

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

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))
    
class AnalysisFramesConstructor:
    """ AnalysisFramesConstructor build time-series analysis frames 
    from an input signal """

    def __init__(self,signalData,frameParams):
        """ Constructor for AnalysisFramesConstructor using AnalysisFramesParameters """
        self._signalData    = signalData
        self._params        = frameParams
        self._window        = None

    def __del__(self):
        """ Destructor for AnalysisFramesConstructor Instance """
        self._signalData    = None
        self._params        = None

    # Getters and Setters

    def getSamplesPerFrame(self) -> int:
        """ Get the Number of Samples in Each Frame """
        return self._params._samplesPerFrame

    def getSamplesOverlap(self) -> int:
        """ Get the Number of Overlap Samples in Each Frame """
        return self._params._samplesOverlap

    def getSizeHeadPad(self) -> int:
        """ Get the Size of the Head Pad """
        return self._params._padHead

    def getSizeTailPad(self) -> int:
        """ Get the size of the Tail Pad """
        return self._params._padTail

    def getMaxNumFrames(self) -> int:
        """ Get the Max Number of Frames to Use """
        return self._params._maxFrames

    def getNumFramesInUse(self) -> int:
        """ Get the Number of Frames Currently in use """
        return self._params._framesInUse

    def getSampleStep(self) -> int:
        """ Get the Sample Step Between adjacent analysis frames """
        return (self._params._samplesPerFrame - self._params._samplesOverlap)

    def getTotalFrameSize(self) -> int:
        """ Get total Size of Each Frame including padding """
        result = 0
        result += self._params._padHead
        result += self._params._samplesPerFrame 
        result += self._params._padTail
        return result

    def getFramesShape(self):
        """ Get the Shape of the Analysis Frames Matrix """
        return ( self.getMaxNumFrames(), self.getTotalFrameSize(), )
    
    # Public Interface

    def call(self):
        """ Convert Signal to Analysis Frames """
        self._signalData.AnalysisFramesTime = \
            np.zeros(shape=self.getFramesShape(),dtype=np.float32)
        self.buildAnalysisFrames()

        # Return the New Signal Data Object
        return self._signalData

    # Private Interface

    def buildAnalysisFrames(self):
        """ Construct Analysis Frames """
        startIndex = 0
        numSamples = self._signalData.Waveform.shape[0]
        padHead = self.getSizeHeadPad()
        frameSize = self.getSamplesPerFrame()

        # Copy all of the frames
        for i in range(self.getMaxNumFrames()):
        
            # Copy slice to padded row
            np.copyto(
                dst=self._signalData.AnalysisFramesTime[i,padHead:padHead + frameSize],
                src=self._signalData.Waveform[startIndex:startIndex + frameSize],
                casting='no')
            
            # Increment
            startIndex += self.getSampleStep()
            self._params._framesInUse += 1

            if (startIndex + frameSize > numSamples):
                # That was just the last frame
                break
      
        return self

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class BatchData:
    """ Class To Hold Data for Each Batch of Samples """
        
    def __init__(self,batchIndex,numSamples,numFeatures):
        """ Constructor for BatchDataInstance """
        self._batchIndex        = batchIndex
        self._numSamplesExpt    = numSamples
        self._numSamplesRead    = 0
        self._means             = np.zeros(shape=(numFeatures),dtype=float)
        self._variances         = np.zeros(shape=(numFeatures),dtype=float)
        
    def __del__(self):
        """ Destructor for BatchData Instance """
        pass

    # Getters and Setters

    def getBatchIndex(self) -> int:
        """ Get the Index of this Batch """
        return self._batchIndex

    def getExpectedNumSamples(self):
        """ Get the number of samples expected to process """
        return self._numSamplesExpt

    def getActualNumSamples(self):
        """ Get the number of samples actually processed """
        return self._numSamplesRead

    def getMeans(self):
        """ Get the Average of Each Feature """
        return self._means

    def getVariances(self):
        """ Get the Variance of each Feature """
        return self._variances

    def incrementNumSamplesRead(self,amount):
        """ Increment the Number of Samples Read in this Batch """
        self._numSamplesRead += amount
        return self

    # Public Interface

    def export(self,path):
        """ Write out the Batch's Data to a specified file """
        return self

    # Private Interface

class WindowFunctions:
    """ Static Class to Hold All Window Functions """

    PadHead = 1024
    PadTail = 2048
    windowSize = lambda x,y,z : x + y + z

    def __init__(self):
        """ Dummy Constructor - Raises Error """
        errMsg = str(self.__class__) + " is a static class, cannot make instance"
        raise RuntimeError(errMsg)

    @staticmethod
    def getWindowSize(*items):
        """ Get Window Size """
        val = 0
        for item in items:
            val += item
        return val

    @staticmethod
    def getHanning(numSamples,headPad=None,tailPad=None):
        """ Get a Hanning Window of the Specified Size """
        if (headPad is None):
            headPad = WindowFunctions.PadHead
        if (tailPad is None):
            tailPad = WindowFunctions.PadTail
        window = np.zeros(
            shape=(WindowFunctions.windowSize(numSamples,headPad,tailPad),),
            dtype=np.float32)
        window[headPad:tailPad] = scisig.windows.hann(numSamples)
        return window

