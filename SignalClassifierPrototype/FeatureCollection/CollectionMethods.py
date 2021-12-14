"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureCollection  
File:           CollectionMethods.py
 
Author:         Landon Buell
Date:           December 2021
"""

            #### IMPORTS ####

import os
import sys
import numpy as np
from numpy.core.fromnumeric import partition

import Administrative
import Structural

            #### CLASS DEFINIIONS ####

class CollectionMethod:
    """
    Abstract Base Class for All Collection Methods to Be Queued
    """

    def __init__(self,name,param):
        """ Constructor for CollectionMethod Base Class """
        self._methodName    = name
        self._parameter     = param

    def __del__(self):
        """ Destructor for CollectionMethod Base Class """
        pass

    # Getters and Setters

    def getMethodName(self) -> str:
        """ Get the Name of this Collection method """
        return str(self.__class__)

    def getReturnSize(self) -> int:
        """ Get the Number of Features that we expect to Return """
        return self._parameter

    # Public Interface

    def invoke(self,signalData,*args):
        """ Run this Collection method """
        if (Administrative.CollectionApplicationProtoype.AppInstance.getSettings().getVerbose() > 1):
            msg = "\t\tInvoking " + self.getMethodName()
            Administrative.CollectionApplicationProtoype.AppInstance.logMessage(msg)
        return np.ones(shape=(self.getReturnSize(),),dtype=np.float32) * -1

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        if (self._parameter <= 0):
            # Must be 1 or More
            errMsg = "Parameter must be greater than or equal to 1!"
            raise ValueError(errMsg)
        return True

    # Magic Methods

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + hex(id(self))

class TimeDomainEnvelopPartitions (CollectionMethod):
    """ Computes the Time-Domain-Envelope by breaking Signal into partitions """

    def __init__(self,numPartitions):
        """ Constructor for TimeDomainEnvelopPartitions Instance """
        super().__init__("TimeDomainEnvelopPartitions",numPartitions)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TimeDomainEnvelopPartitions Instance """
        pass

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)  
        sizeOfPartition = signalData.Waveform.shape[0] // self._parameter
        # Iterate Through Each Parition
        startIndex = 0
        for i in range(self._parameter):    
            part = signalData.Waveform[ startIndex : startIndex + sizeOfPartition]
            result[i] = np.sum(np.abs(part**2),dtype=np.float32)
            startIndex += sizeOfPartition
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "Signal.Samples must not be None"
            raise ValueError(errMsg)
        if (signalData.Waveform.shape[0] < 2* self._parameter):
            errMsg = "Signal.Samples is too small to compute TDE"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """ 
        super().validateParameter()
        if (self._parameter < 2 or self._parameter > 32):
            # Param should be greater than 1 and less than 33
            errMsg = "numParitions should be greater than 2 and less than 33"
        return True

class TimeDomainEnvelopFrames(CollectionMethod):
    """ Computes the TimeDomainEnvelop of Each Time-Series Analysis Frame """

    def __init__(self,maxFrames=256):
        """ Constructor for TimeDomainEnvelopFrames Instance """
        super().__init__("TimeDomainEnvelopFrames",maxFrames)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TimeDomainEnvelopFrames Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesTime is None):
            errMsg = "Signal.SaAnalysisFramesTimemples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class PercentFramesAboveEnergyThreshold(CollectionMethod):
    """
    Compute the Number of Frames with energy above threshold% of Maximum energy
    """

    def __init__(self,threshold):
        """ Constructor for PercentFramesEnergyAboveThreshold Instance """
        super().__init__("PercentFramesEnergyAboveThreshold",1)
        self._thresholdFactor = threshold
        self.validateParameter()

    def __del__(self):
        """ Destructor for PercentFramesEnergyAboveThreshold Instance """
        pass

    # Getters and Setters

    def getThresholdFactor(self):
        """ Get the Threshold Factor for this instance """
        return self._thresholdFactor

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesTime is None):
            errMsg = "Signal.SaAnalysisFramesTimemples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsPerTime(CollectionMethod):
    """
    Compute the total number of zero crossings normalized by signal length
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsPerTime Instance """
        super().__init__("ZeroCrossingsPerTime",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsPerTime Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "Signal.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesMean(CollectionMethod):
    """
    Compute the average number of zero crossings over all analysis frames
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsFramesAverage Instance """
        super().__init__("ZeroCrossingsFramesAverage",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesAverage Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.ZeroCrossingFrames is None):
            errMsg = "Signal.ZeroCrossingFrames must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesVariance(CollectionMethod):
    """
    Compute the variance of zero crossings over all analysis frames
    """

    def __init__(self,param=1):
        """ Constructor for ZeroCrossingsFramesVariance Instance """
        super().__init__("ZeroCrossingsFramesVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesVariance Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.ZeroCrossingFrames is None):
            errMsg = "Signal.ZeroCrossingFrames must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class ZeroCrossingsFramesDiffMinMax(CollectionMethod):
    """
    Compute the difference of the min and max of zero crossings 
    over all analysis frames
    """

    def __init__(self,param):
        """ Constructor for ZeroCrossingsFramesDiffMinMax Instance """
        super().__init__("ZeroCrossingsFramesDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for ZeroCrossingsFramesDiffMinMax Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.ZeroCrossingFrames is None):
            errMsg = "Signal.ZeroCrossingFrames must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class TemporalCenterOfMassLinear(CollectionMethod):
    """
    Compute the Temporal Center of Mass, weighted linearly
    """

    def __init__(self,param):
        """ Constructor for TemporalCenterOfMassLinear Instance """
        super().__init__("TemporalCenterOfMassLinear",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TemporalCenterOfMassLinear Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "Signal.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class TemportalCenterOfMassQuadratic(CollectionMethod):
    """
    Compute the Temporal Center of Mass, weighted Quadratically
    """

    def __init__(self,param):
        """ Constructor for TemportalCenterOfMassQuadratic Instance """
        super().__init__("TemportalCenterOfMassQuadratic",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TemportalCenterOfMassQuadratic Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "Signal.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficients(CollectionMethod):
    """
    Compute the First k Auto-CorrelationCoefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for AutoCorrelationCoefficients Instance """
        super().__init__("AutoCorrelationCoefficients",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficients Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "Signal.Samples must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsMean(CollectionMethod):
    """
    Compute the mean of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsMean Instance """
        super().__init__("AutoCorrelationCoefficientsMean",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsMean Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "Signal.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsVariance(CollectionMethod):
    """
    Compute the variance of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsVariance Instance """
        super().__init__("AutoCorrelationCoefficientsVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsVariance Instances """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "Signal.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class AutoCorrelationCoefficientsDiffMinMax(CollectionMethod):
    """
    Compute the Different of min and max of the first Auto-Correlation-Coefficients
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsDiffMinMax v """
        super().__init__("AutoCorrelationCoefficientsDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsDiffMinMax Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "Signal.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FreqDomainEnvelopPartition(CollectionMethod):
    """
    Compute the Frequency Domain Envelop of each frame and average down all of them
    """

    def __init__(self,param):
        """ Constructor for AutoCorrelationCoefficientsDiffMinMax Base Class """
        super().__init__("AutoCorrelationCoefficientsDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsDiffMinMax Base Class """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "Signal.AutoCorrelationCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FreqDomainEnvelopFrames(CollectionMethod):
    """ 
    Computes the Frequency Domain Envelope of each Freq Analysis Frame
    and then average each partition across all frames    
    """

    def __init__(self,numPartitions):
        """ Constructor for TimeDomainEnvelopFrames Instance """
        super().__init__("TimeDomainEnvelopFrames",numPartitions)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TimeDomainEnvelopFrames Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "Signal.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FrequencyCenterOfMassLinear(CollectionMethod):
    """
    Compute the Frequency Center of Mass over all frames weighted linearly
    """

    def __init__(self,param):
        """ Constructor for FrequencyCenterOfMassLinear Instance """
        super().__init__("FrequencyCenterOfMassLinear",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for FrequencyCenterOfMassLinear Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "Signal.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FrequencyCenterOfMassQuadratic(CollectionMethod):
    """
    Compute the frequency Center of Mass over all frames, weighted Quadratically
    """

    def __init__(self,param):
        """ Constructor for FrequencyCenterOfMassQuadratic Instance """
        super().__init__("FrequencyCenterOfMassQuadratic",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for FrequencyCenterOfMassQuadratic Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "Signal.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFrequencyCempstrumCoeffs(CollectionMethod):
    """
    Compute K Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for MelFrequencyCempstrumCoeffs Instance """
        super().__init__("MelFrequencyCempstrumCoeffs",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffs Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "Signal.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFrequencyCempstrumCoeffsMean(CollectionMethod):
    """
    Compute Average of Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,numCoeffs):
        """ Constructor for MelFrequencyCempstrumCoeffsMean Instance """
        super().__init__("MelFrequencyCempstrumCoeffsMean",numCoeffs)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsMean Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "Signal.MelFreqCepstrumCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFrequencyCempstrumCoeffsVariance(CollectionMethod):
    """
    Compute variance of Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,param):
        """ Constructor for MelFrequencyCempstrumCoeffsVariance Instance """
        super().__init__("MelFrequencyCempstrumCoeffsVariance",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsVariance Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "Signal.MelFreqCepstrumCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class MelFrequencyCempstrumCoeffsDiffMinMax(CollectionMethod):
    """
    Compute difference of min and max Mel Frequency Cepstrum Coefficients
    """

    def __init__(self,param):
        """ Constructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        super().__init__("MelFrequencyCempstrumCoeffsDiffMinMax",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for MelFrequencyCempstrumCoeffsDiffMinMax Instance """
        pass

    # Public Interface

    def invoke(self, signal, *args):
        """ Run this Collection method """
        result = super().invoke(signal)   
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "Signal.MelFreqCepstrumCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True