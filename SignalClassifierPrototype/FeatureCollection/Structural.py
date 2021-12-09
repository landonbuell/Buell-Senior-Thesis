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

class FeatureVector:
    """ Class to Hold Feature Data for a single Sample """

    def __init__(self,sampleShape,label=-1):
        """ Constructor for FeatureVector Instance """
        self._sampleShape   = sampleShape
        self._label         = label
        self._data          = np.zeros(shape=sampleShape,dtype=np.float32)

    def __del__(self):
        """ Destructor for FeatureVector Instance """
        self.clearData()

    # Getters and Setters

    def getShape(self):
        """ Get the Shape of this Sample """
        return self._sampleShape

    def getLabel(self):
        """ Get the Target Label """
        return self._label

    def setLabel(self,x):
        """ Set the Target Label """
        self._label = x
        return self

    def getData(self):
        """ Get the underlying Array """
        return self._data

    def setData(self,x,enforceShape=True):
        """ Set the Underlying Array, optionally chanign shape """
        if (enforceShape == True):
            assert(x.shape == self.getShape())
            self._data = x
        else:
            self._sampleShape = x.shape
            self._data = x
        return self

    # Public Interface

    def clearData(self):
        """ Clear All Entries in this Array """
        self._label         = -1
        self._data          = np.zeros(shape=self._sampleShape,dtype=np.float32)
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
        self._tgts          = np.zeros(shape=numSamples,dtype=np.int16)

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

    def getData(self):
        """ Get Design Matrix as an Array """
        return self._data

    def setData(self,x):
        """ Set Design Matrix is an Array """
        self._numSamples = x.shape[0]
        self._sampleShape = x.shape[1:]
        self._data = x
        return self

    def getLabels(self):
        """ Get the Labels as an Array """
        return self._tgts

    def setLabels(self,x):
        """ Set the Labels as an Array """
        self._tgts = x
        return self

    def getUniqueClasses(self):
        """ Get An Array of the unique classes """
        return np.unique(self._tgts)

    def getNumClasses(self):
        """ Get the Number of classes in the data set """
        return np.max(self._tgt)

    # public Interface

    def serialize(self,path=None):
        """ Write this design matrix out to a file """
        writer = DesignMatrix.DesignMatrixSerializer(path)
        writer.call(self)
        return self

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data          = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts          = np.zeros(shape=self.getNumSamples(),dtype=np.int16)
        return self

    @staticmethod
    def encodeOneHot(targets,numClasses):
        """ Get a One-Hot-Encoded Array of targets """
        numSamples = targets.shape[-1]
        result = np.zeros(shape=(numSamples,numClasses),dtype=np.int16)   
        for i in range(numSamples):
            tgt = targets[i]
            result[i,tgt] = 1
        return result

    # Private Interface

    class DesignMatrixSerializer:
        """ Private Class to Serialize a DesignMatrixInstance """
        
        def __init__(self,outputPath):
            """ Constructor for DesignMatrixSerializer Instance """
            self._outputPath = outputPath
            self._fileHandle = None

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            if (self._fileHandle is not None):
                self._fileHandle.close()

        def call(self,designMatrix):
            """ Run the Serializer """
            numSamples = designMatrix.getNumSamples()
            X = designMatrix.getData()
            Y = designMatrix.getLabels()
            # Create + Write to output
            self._fileHandle = open(self._outputPath,"wb")
            for i in range(numSamples):
                tgt = X[i].astype(np.int32).tobytes()
                row = Y[i].flatten().tobytes()
                self._fileHandle.write( tgt )
                self._fileHandle.write( row )
            # Close + Return
            self._fileHandle.close()
            return self

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
        # Make a Feature Vector + Return it
        featureVector = FeatureVector(self._sampleShape,self._tgts[key])
        featureVector.setData(self._data[key])
        return featureVector

    def __setitem__(self,key,value):
        """ Set the Item at the Index """
        assert(value.getShape() == self._sampleShape)
        self._tgts[key] = value.getLabel()
        self._data[key] = value.getData()
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

    def getMeans(self):
        """ Get the Average of Each Feature """
        return self._means

    def getVariances(self):
        """ Get the Variance of each Feature """
        return self._variances

    # Public Interface


    # Private Interface
