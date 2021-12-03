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
from typing_extensions import runtime
import numpy as np

import scipy.io.wavfile as sciowav
import scipy.fftpack as fftpac

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
        self._frameEnergyTime       = None
        self._frameEnergyFreq       = None

    def __del__(self):
        """ Destructor for SignalData Instance """
        self._samples               = None
        self._analysisFramesTime    = None
        self._analysisFramesFreq    = None
        self._melFreqCepstrumCoeffs = None
        self._frameEnergyTime       = None
        self._frameEnergyFreq       = None
       
    # Getters and Setters

    def getSampleRate(self):
        """ Get the Sample Rate """
        return self._sampleRate

    def getSampleSpace(self):
        """ Get the Sample Spacing """
        return (1/self._sampleRate)

    @property
    def Samples(self):
        """ Access Time-Series Samples """
        return self._samples

    @property
    def AnalysisFramesTime(self):
        """ Access Time-Series Analysis Frames """
        return self._analysisFramesTime