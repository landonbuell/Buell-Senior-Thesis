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
            result[i] = np.sum((part**2),dtype=np.float32)
            startIndex += sizeOfPartition
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Waveform must not be None"
            raise ValueError(errMsg)
        if (signalData.Waveform.shape[0] < 2* self._parameter):
            errMsg = "signalData.Waveform is too small to compute TDE"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """ 
        super().validateParameter()
        if (self._parameter < 2 or self._parameter > 32):
            # Param should be greater than 1 and less than 33
            errMsg = "numParitions should be greater than 2 and less than 33"
            raise ValueError(errMsg)
        return True

class TimeDomainEnvelopFrames(CollectionMethod):
    """ Computes the TimeDomainEnvelop of Each Time-Series Analysis Frame """

    def __init__(self,startFrame=0,endFrame=256,skip=1):
        """ Constructor for TimeDomainEnvelopFrames Instance """
        numFrames = int(endFrame - startFrame) // skip
        super().__init__("TimeDomainEnvelopFrames",numFrames)
        self.validateParameter()
        self._numFrames     = numFrames
        self._start         = startFrame
        self._stop          = endFrame
        self._step          = skip

    def __del__(self):
        """ Destructor for TimeDomainEnvelopFrames Instance """
        pass

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData) 
        idx = 0
        for i in range(self._start,self._stop,self._step):
            result[idx] = signalData.FrameEnergyTime[i]
            idx += 1
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameEnergyTime is None):
            # Make the Frame Energies
            signalData.makeFrameEnergiesTime()
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        # Get Max Frame Energy + Find Threshold to beat
        maxEnergy = np.max(signalData.FrameEnergyTime)
        threshold = maxEnergy * self.getThresholdFactor()
        numFrames = 0       # number of frames above the threshold
        totFrames = signalData.FrameEnergyTime.shape[0]

        # Iterate through the Frame Energies
        for item in signalData.FrameEnergyTime:
            if (item > threshold):
                # Meets the energy criteria
                numFrames += 1

        # Get Number of Frames as a percentage
        result[0] = (numFrames / totFrames)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameEnergyTime is None):
            # Make the Frame Energies
            signalData.makeFrameEnergiesTime()
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)  
        
        numSamples = signalData.getNumSamples()
        sign = np.sign(signalData.Waveform)
        ZXR = 0

        # Iterate through Sampeles
        for i in range(1,numSamples):
            ZXR += np.abs(sign[i] - sign[i-1])
        result[0] = ZXR / 2
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Waveform must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)  
        result[0] = np.mean(signalData.FrameZeroCrossings)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)
        result[0] = np.var(signalData.FrameZeroCrossings)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData) 
        minVal = np.min(signalData.FrameZeroCrossings)
        maxVal = np.max(signalData.FrameZeroCrossings)
        result[0] = maxVal - minVal
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.FrameZeroCrossings is None):
            signalData.makeZeroCrossingRate()
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class TemporalCenterOfMass(CollectionMethod):
    """
    Compute the Temporal Center of Mass, weighted Quadratically
    """

    def __init__(self,param):
        """ Constructor for TemporalCenterOfMass Instance """
        super().__init__("TemportalCenterOfMassQuadratic",param)
        self.validateParameter()

    def __del__(self):
        """ Destructor for TemporalCenterOfMass Instance """
        super().__del__()

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        # Compute Total Mass + Weights
        massTotal = np.sum(signalData.Waveform)
        weights = np.arange(0,signalData.getNumSamples())**(self._parameter)
        # Compute Center of Mass (By Weights)
        massCenter = np.dot(weights,signalData.Waveform);
        massCenter /= massTotal
        massCenter /= signalData.getNumSamples()

        # Apply Result + Return 
        result[0] = massCenter
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Samples must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        #Check is ACC's exist - make them if not
        if (signalData.AutoCorrelationCoeffs is None):
            signalData.makeAutoCorrelationCoeffs(self._parameter)

        # Copy the ACC's the the result + Return
        np.copyto(result,signalData.AutoCorrelationCoeffs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.Waveform is None):
            errMsg = "signalData.Samples must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        # Get the Average of the AutoCorrelation Coefficients
        result[0] = np.mean(signalData.AutoCorrelationCoeffs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData) 
        
        # Compute the Variance
        result[0] = np.var(signalData.AutoCorrelationCoeffs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData) 
        # Compute Difference between Min and Max
        minVal = np.min(signalData.AutoCorrelationCoeffs)
        maxVal = np.max(signalData.AutoCorrelationCoeffs)
        result[0] = maxVal - minVal
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
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
        super().__init__("FreqDomainEnvelopPartition",1)
        self.validateParameter()

    def __del__(self):
        """ Destructor for AutoCorrelationCoefficientsDiffMinMax Base Class """
        pass

    # Public Interface

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData);
        result = super().invoke(signalData)   
        raise NotImplementedError(str(self.__class__) + " is not implemented")
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AutoCorrelationCoeffs is None):
            errMsg = "signalData.AutoCorrelationCoeffs must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   
        raise NotImplementedError(str(self.__class__) + " is not implemented")

        sizeOfPartition = signalData.AnalysisFramesFreq.shape[-1] // self._parameter
        # Iterate Through Each Parition
        startIndex = 0
        for i in range(self._parameter):    
            part = signalData.AnalysisFramesFreq[:, startIndex : startIndex + sizeOfPartition]
            part = np.sum(part**2,axis=1,dtype=np.float32)
            result[i] = np.mean(part)
            startIndex += sizeOfPartition
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "signalData.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

class FrequencyCenterOfMass(CollectionMethod):
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        # Compute Mass of Each Frame
        sizeOfFrame = signalData.AnalysisFramesFreq.shape[1]
        massTotal = np.sum(signalData.AnalysisFramesFreq,axis=-1)
        weights = np.arange(0,sizeOfFrame,1)**(self._parameter)
        centerOfMasses = np.matmul(signalData.AnalysisFramesFreq,weights)
        centerOfMasses /= massTotal

        # Add the Average of all frames, and put into result
        result[0] = np.mean(centerOfMasses)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "signalData.AnalysisFramesFreq must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   

        # Check if We have MFCC's - Create if we don't
        if (signalData.MelFreqCepstrumCoeffs is None):
            signalData.makeMelFreqCepstrumCoeffs(self._parameter)
        avgMFCCs = np.mean(signalData.MelFreqCepstrumCoeffs,axis=0)

        # Copy to result + return
        np.copyto(result,avgMFCCs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.AnalysisFramesFreq is None):
            errMsg = "signalData.AnalysisFramesFreq must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True

    # Static Interface

    @staticmethod
    def melsToHertz(freqMels):
        """ Cast Mels Samples to Hertz Samples """
        return 700 * ( 10** (freqMels / 2595) -1 )

    @staticmethod
    def hertzToMels(freqHz):
        """ Cast Hertz Samples to Mels Samples """
        return 2595 * np.log10(1 + freqHz / 700)

    @staticmethod
    def melFilterBanks(numFilters,sampleRate=44100):
        """ Build the Mel-Filter Bank Arrays """
        frameParams = Administrative.CollectionApplicationProtoype.AppInstance.getRundataManager().getFrameParams()
        freqBoundsHz = frameParams.getFreqBoundHz()
        freqBoundsMels = MelFrequencyCempstrumCoeffs.hertzToMels(freqBoundsHz)
        numSamplesTime = frameParams.getTotalTimeFrameSize()       

        freqAxisMels = np.linspace(freqBoundsMels[0],freqBoundsMels[1],numFilters+2)
        freqAxisHz = MelFrequencyCempstrumCoeffs.melsToHertz(freqAxisMels)
        bins = np.floor((numSamplesTime+1)*freqAxisHz/sampleRate)
        filterBanks = np.zeros(shape=(numFilters,numSamplesTime),dtype=np.float32)

        # Iterate through filters
        for i in range (1,numFilters + 1,1):
            freqLeft = int(bins[i-1])
            freqCenter = int(bins[i])
            freqRight = int(bins[i+1])

            for j in range(freqLeft,freqCenter):
                filterBanks[i-1,j] = (j - freqLeft) / (freqCenter - freqLeft)
            for j in range(freqCenter,freqRight):
                filterBanks[i-1,j] = (freqRight - j) / (freqRight - freqCenter)

        # Crop to Subset of Frequency Space
        numSamplesFreq = frameParams.getFreqFramesShape()[1]
        filterBanks = filterBanks[:,:numSamplesFreq]
        return filterBanks


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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   
        # Compute Mean of MFCC's
        avgMFCCs = np.mean(signalData.MelFreqCepstrumCoeffs,axis=0)
        result[0] = np.mean(avgMFCCs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "signalData.MelFreqCepstrumCoeffs must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   
        # Compute Variance of MFCC's
        avgMFCCs = np.mean(signalData.MelFreqCepstrumCoeffs,axis=0)
        result[0] = np.var(avgMFCCs)
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "signalData.MelFreqCepstrumCoeffs must not be None"
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

    def invoke(self, signalData, *args):
        """ Run this Collection method """
        self.validateInputSignal(signalData)
        result = super().invoke(signalData)   
        # Compute Diff of min and max of MFCC's
        avgMFCCs = np.mean(signalData.MelFreqCepstrumCoeffs,axis=0)
        minVal = np.min(avgMFCCs)
        maxVal = np.max(avgMFCCs)
        result[0] = maxVal - minVal
        return result

    # Protected Interface

    def validateInputSignal(self,signalData):
        """ Validate Input Signal Everything that we need """
        if (signalData.MelFreqCepstrumCoeffs is None):
            errMsg = "signalData.MelFreqCepstrumCoeffs must not be None"
            raise ValueError(errMsg)
        return True

    def validateParameter(self):
        """ Validate that Parameter Values Makes Sense """
        super().validateParameter()
        return True