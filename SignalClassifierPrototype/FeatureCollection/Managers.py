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
import pandas as pd

import Administrative
import CollectionMethods
import Structural

import CommonStructures

        #### CLASS DEFINITIONS ####

class Manager:
    """
    Manager is an Abstract Base Class in which all managers inherit from
    """

    def __init__(self):
        """ Constructor for Manager Base Class """
        self.logConstruction()

    def __del__(self):
        """ Destructor for Manager Base Class """
        self.logDestruction()

    # Getters and Setters

    def getRuntimeSettings(self):
        """ Get a reference to the Runtime Settings from the AppInstance """
        return Administrative.CollectionApplicationProtoype.AppInstance.getSettings()

    def getSampleManager(self):
        """ Get a reference to the Sample Manager form the AppInstance """
        return Administrative.CollectionApplicationProtoype.AppInstance.getSampleManager()

    def getCollectionManager(self):
        """ Get a reference to the collection Manager from the AppInstance """
        return Administrative.CollectionApplicationProtoype.AppInstance.getCollectionManager()

    def getRundataManager(self):
        """ Get a reference to the Rundata Manager from the the AppInstance """
        return Administrative.CollectionApplicationProtoype.AppInstance.getRundataManager()

    # Public Interface

    def build(self):
        """ Initialize all Paramters for this Manager """
        self.logBuild()
        return self

    def call(self):
        """ Run the Execution of this Manager """
        self.logExecution()
        return self

    def clean(self):
        """ Cleanup the Manager """
        self.logCleanup()
        return self

    def logMessageInterface(self,msg,timeStamp=True):
        """ Simplified Interface for Logging Message via the CollectionApplicationPrototype """
        Administrative.CollectionApplicationProtoype.AppInstance.logMessage(msg,timeStamp)
        return None

    # Protected Interface

    def describe(self):
        """ Log Description of the Current State of this Instance """
        return self

    def logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logBuild(self):
        """ Log the Initialization of the instance """
        msg = "Initializing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logExecution(self):
        """ Log the Initialization of the instance """
        msg = "Executing " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logCleanup(self):
        """ Log the Initialization of the instance """
        msg = "Cleaning " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    def logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying " + str(self.__class__) + " Instance..."
        self.logMessageInterface(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))


class SampleManager (Manager):
    """ SampleManager collects and organizes all data samples """

    def __init__(self):
        """ Constructor for SampleManager Instance """
        super().__init__()
        self._sampleDataBase    = np.array([],dtype=object)
        self._labelDictionary   = dict({})
        self._classCounter      = None
        self._batchSizes        = None
        self._sampleIndex       = 0

    def __del__(self):
        """ Destructor for SampleManager Instance """
        super().__del__()
        
    # Getters and Setters

    def getTargetStr(self,targetInt) -> str:
        """ Get Corresponding Target String from Dictionary """
        return self._labelDictionary.get(targetInt)

    def setTarget(self,intTgt,strTgt):
        """ Update Label Dictionary w/ int:str pair """
        self._labelDictionary.update({intTgt:strTgt})
        return self

    def getSample(self,idx):
        """ Get Sample From Database at Index """
        return self._sampleDataBase[idx]

    def setSample(self,idx,sample):
        """ Set Sample to Database at Index """
        self._sampleDataBase[idx] = sample
        return self

    def getBatchSizes(self):
        """ Get Array of Each Batch Size """
        return self._batchSizes

    def getNumClasses(self) -> int:
        """ Get the Number of Classes by entries in the Dictionary """
        return len(self._labelDictionary)

    def getNumSamples(self) -> int:
        """ Get the Total Number of Samples """
        return self._sampleDataBase.shape[0]

    def getNumBatches(self) -> int:
        """ Get the Number of Batches in this Run """
        return self._batchSizes.shape[0]

    def getSizeOfBatch(self, batchIndex: int) -> int:
        """ Get the Size of the i-th batch """
        if (batchIndex >= self.getNumBatches()):
            errMsg = "Batch Index is out of range"
            raise ValueError(errMsg)
        return self._batchSizes[batchIndex]

    def getNextSample(self):
        """ Get the Sample Pointed to by the Index """
        if (self._sampleIndex >= self.getNumSamples()):
            result = None
        else:
            result = self._sampleDataBase[self._sampleIndex]
            self._sampleDataBase[self._sampleIndex] = 0
            self._sampleIndex += 1
        return result

    # Public Interface

    def build(self):
        """ Gather + Organize all Audio Samples """
        super().build()

        self.readInputFiles()
        self.createSizeOfEachBatch()
        self.initClassCounter()
        self.describe()

        return self

    def initClassCounter(self):
        """ Count the Number of Samples in Each Class """
        self._classCounter = np.zeros(shape=(self.getNumClasses(),),dtype=np.uint32)
        return self

    def describe(self):
        """ Log description of state of this instance """

        # Basic Info
        messages = [
            "Total samples: {0}".format(len(self)),
            "Number of classes: {0}".format(self.getNumClasses()),
            "Number of batches: {0}".format(self.getNumBatches())
            ]
        for msg in messages:
            # Log Each String as a Message
            self.logMessageInterface(msg)

        # Log the Label Dictionary
        for (key,val) in self._labelDictionary.items():
            msg = "{0:<32}\t{1:<16}\t{2:<32}".format(" ",key,val)
            self.logMessageInterface(msg,False)

        return self

    def createBatch(self,batchIndex: int):
        """ Get an Array of Samples for the Next Batch """
        # Create the Batch Subset
        batchSize = self.getSizeOfBatch(batchIndex)
        indexStart = batchIndex * batchSize
        batch = np.empty(shape=(batchSize,),dtype=object)
        
        # Populate Batch w/ Entries from Database
        for i in range(batchSize):
            batch[i] = self.getNextSample()

        return batch

    def updateClassCounter(self,targetInt: int):
        """ Update Class Counter w/ sample target """
        pass

    # Private Interface

    def readInputFiles(self):
        """ Read Through All Input Files and Add to Sample Database """
        inputFiles = self.getRuntimeSettings().getInputPaths()
        samplesInFile = None
        
        # Visit Each Input File + Get All Samples
        for path in inputFiles:
            # Log this File
            msg = "Reading samples from file: {0}".format(path)
            self.logMessageInterface(msg)
            # Get the Data
            samplesInFile = self.createSamplesFromFile(path)
            self._sampleDataBase = np.append(self._sampleDataBase,samplesInFile)
            # Log Number of Samples
            msg = "\tFound {0} samples".format(samplesInFile.shape[0])
            self.logMessageInterface(msg)

        return self

    def createSamplesFromFile(self,filePath):
        """ Read a file, and return an array of samples from it """
        frame = pd.read_csv(filePath,index_col=False)
        frame = frame.to_numpy()
        sampleArray = np.empty(shape=(frame.shape[0],),dtype=object)

        # Visit Each Sample in the Row
        for i,row in enumerate(frame):
            # Get Params from the row
            samplePath  = row[0]
            tgtInt      = int(row[1])
            tgtStr      = row[2]
            # Create the SampleIO Instance + Update Int -> Str Map
            sample = Structural.SampleIO(samplePath,tgtInt,tgtStr)
            self.setTarget(tgtInt,tgtStr)
            # Add the Sample
            sampleArray[i] = sample

        return sampleArray

    def createSizeOfEachBatch(self):
        """ Build a List for the Size of Each Batch """
        standardBatchSize = Administrative.CollectionApplicationProtoype.AppInstance.getSettings().getBatchSize()
        numSamples = self._sampleDataBase.shape[0]
        numBatches = (numSamples // standardBatchSize)
        allBatchSizes = np.ones(shape=(numBatches,),dtype=int) * standardBatchSize
        extraSamples =  (numSamples % standardBatchSize)
        # Computer the Number of Batches (Include )
        if (extraSamples != 0):
            # There are "Extra" Samples
            allBatchSizes = np.append(allBatchSizes,extraSamples)
        self._batchSizes = allBatchSizes
        return self
          
    # Magic Methods

    def __len__(self):
        """ Overload Length Operator """
        return self._sampleDataBase.shape[0]

    
class CollectionManager (Manager):
    """ CollectionManager organizes all Features Methods """

    def __init__(self):
        """ Constructor for CollectionManager Instance """
        super().__init__()
        self._batchIndex        = 0
        self._batchQueue        = np.array([],dtype=object)
        self._methodQueue       = np.array([],dtype=object)    
        self._designMatrixA     = None
        self._designMatrixB     = None
        
    def __del__(self):
        """ Destructor for CollectionManager Instance """
        self._batchQueue        = None
        self._methodQueue       = None
        self._designMatrixA      = None
        self._framesParameters  = None
        self._framesContructor  = None
        super().__del__()

    # Getters and Setters

    def getBatchIndex(self) -> int:
        """ Get the Current Batch Index """
        return self._batchIndex

    def getBatchQueue(self):
        """ Get the Current Batch Queue of Audio Files """
        return self._batchQueue

    def getMethodQueue(self):
        """ Get the Method Queue for the Collector """
        return self._methodQueue

    def getDesignMatrixA(self):
        """ Get the Design Matrix A"""
        return self._designMatrixA

    def getDesignMatrixB(self):
        """ Get the Design Matrix B"""
        return self._designMatrixB

    def getNumFeatures(self):
        """ Compute the Number of Features from the Method Queue """
        numFeatures = 0
        for item in self._methodQueue:
            if (item == 0):
                continue
            numFeatures += item.getReturnSize()
        return numFeatures

    # Public Interface

    def build(self):
        """ Build All Data for Feature Collection """
        super().build()

        self.createCollectionQueue()
        self.initDesignMatrix()


        return self

    def call(self,batchIndex,batchSize):
        """ The Run the Collection Manager """
        super().call()

        # Log this Batch
        self.logCurrentBatch(batchIndex,batchSize)
        self._batchIndex = batchIndex

        # Build the Design Matrix
        sampleShapeA = (self.getNumFeatures(),)
        sampleShapeB = (1,)
        self._designMatrixA = CommonStructures.DesignMatrix(batchSize,sampleShapeA)
        self._designMatrixB = CommonStructures.DesignMatrix(batchSize,sampleShapeB)

        # Create + Evaluate the Batch
        self.createBatchQueue(batchIndex)
        self.evaluateBatchQueue()            

        # Serialize the Design Matrix
        outputPath = os.path.join(
            Administrative.CollectionApplicationProtoype.AppInstance.getSettings().getOutputPath(),
            "batch{0}.bin".format(batchIndex))

        self._designMatrixA.serialize(outputPath)

        # Compute Meta Data and then Clear
        self._designMatrixA.clearData()

        return self

    # Private Interface

    def createCollectionQueue(self):
        """ Build All Elements in the Collection Queue """
        numEntries = 32
        self._methodQueue = np.zeros(shape=(numEntries,),dtype=object)
        self[0] = CollectionMethods.TimeDomainEnvelopPartitions(8)
        self[1] = CollectionMethods.TimeDomainEnvelopFrames(1)
        self[2] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.1)
        self[3] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.2)
        self[4] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.3)
        self[5] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.4)
        self[6] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.5)
        self[7] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.6)
        self[8] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.7)
        self[9] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.8)
        self[10] = CollectionMethods.PercentFramesAboveEnergyThreshold(0.9)
        self[11] = CollectionMethods.ZeroCrossingsPerTime(1)
        self[12] = CollectionMethods.ZeroCrossingsFramesMean(1)
        self[13] = CollectionMethods.ZeroCrossingsFramesVariance(1)
        self[14] = CollectionMethods.ZeroCrossingsFramesDiffMinMax(1)
        self[15] = CollectionMethods.TemporalCenterOfMassLinear(1)
        self[16] = CollectionMethods.TemportalCenterOfMassQuadratic(1)
        self[17] = CollectionMethods.AutoCorrelationCoefficients(12)
        self[18] = CollectionMethods.AutoCorrelationCoefficientsMean(1)
        self[19] = CollectionMethods.AutoCorrelationCoefficientsVariance(1)
        self[20] = CollectionMethods.AutoCorrelationCoefficientsDiffMinMax(1)
        self[21] = CollectionMethods.FreqDomainEnvelopPartition(12)
        self[22] = CollectionMethods.FreqDomainEnvelopFrames(1)
        self[23] = CollectionMethods.FrequencyCenterOfMassLinear(1)
        self[24] = CollectionMethods.FrequencyCenterOfMassQuadratic(1)
        self[25] = CollectionMethods.MelFrequencyCempstrumCoeffs(12)
        self[26] = CollectionMethods.MelFrequencyCempstrumCoeffsMean(1)
        self[27] = CollectionMethods.MelFrequencyCempstrumCoeffsVariance(1)
        self[28] = CollectionMethods.MelFrequencyCempstrumCoeffsDiffMinMax(1)
        self[29] = 0
        self[30] = 0
        self[31] = 0
        return self

    def initDesignMatrix(self):
        """ Initialize the Design Matrix Instance """
        numSamples = self.getSampleManager().getSizeOfBatch(self._batchIndex)
        shapeA = (self.getNumFeatures(),)
        shapeB = (1,)
        self._designMatrixA = CommonStructures.DesignMatrix(numSamples,shapeA)
        self._designMatrixB = CommonStructures.DesignMatrix(numSamples,shapeB)
        return self

    def createBatchQueue(self,idx):
        """ Create the Current Batch of Samples """      
        self._batchQueue = Administrative.CollectionApplicationProtoype.AppInstance.getSampleManager().createBatch(idx)
        return self

    def evaluateBatchQueue(self):
        """ Iterate through Batch Queue """
        shape = (self.getNumFeatures(),)
        featureVector   = CommonStructures.FeatureVector(shape)
        sampleData      = None
        for idx,sample in enumerate(self._batchQueue):

            # Log this Sample
            self.logCurrentSample(idx,len(self._batchQueue))
            
            # Set the Label + Read the Raw Samples
            featureVector.setLabel(sample.getTargetInt())
            sampleData = sample.readSignal()

            # Construct Analysis Frames
            sampleData = sampleData.makeAnalysisFramesTime(
                self.getRundataManager().getFrameParams() )

            # Use Current Sample to Evaluate the Feature Queue
            self.evaluateMethodQueue(sampleData,featureVector)

            # Add to Batch Design Matrix
            self._designMatrixA[idx] = featureVector
            featureVector.clearData()

        # Update the Batch's Meta Data
        batchData = Structural.BatchData(
            self.getBatchIndex(),
            len(self._batchQueue),
            self.getNumFeatures() )
        self.getRundataManager().addBatchData(batchData)
        sampleData = None
        return self


    def evaluateMethodQueue(self,signalData,featureVector):
        """ Evaluate the Feature Queue """
        featureIndex = 0
        result = None
        for idx,item in enumerate(self._methodQueue):

            if (item == 0):
                # Null Feature
                continue

            # Evalue the current method
            result = item.invoke(signalData)

            # Copy Result to the feature vector
            for i in range(item.getReturnSize()):
                featureVector[featureIndex] = result[i]
                featureIndex += 1
            result = None

        # Sanity Check
        assert(featureIndex == featureVector.getShape()[0])
        return self

    

    def logCurrentBatch(self,index,size):
        """" Log Current Batch w/ Num Samples """
        numBatches = Administrative.CollectionApplicationProtoype.AppInstance.getSampleManager().getNumBatches()
        msg = "Running batch ({0}/{1}), with {2} samples".format(index,numBatches,size)
        self.logMessageInterface(msg)
        return None

    def logCurrentSample(self,index,size):
        """ Log Current Sample in Batch """
        msg = "\tProcessing sample ({0}/{1})".format(index,size)
        self.logMessageInterface(msg)
        return None

    # Magic Methods

    def __len__(self):
        """ Overload Length Operator """
        return self._methodQueue.shape[0]

    def __getitem__(self,key):
        """ Get Item at index """
        return self._methodQueue[key]

    def __setitem__(self,key,val):
        """ Set Item at Index """
        self._methodQueue[key] = val
        return self

    

class RundataManager (Manager):
    """ RundataManager Aggregates all important info from the Collection run process """
    
    def __init__(self):
        """ Constructor for MetadataManager Instance """
        super().__init__()
        self._runInfo           = None
        self._batchDataObjs     = []
        self._frameParams       = None

    def __del__(self):
        """ Destructor for MetadataManager Instance """
        super().__del__()

    # Getters and Setters

    def getRunInfo(self):
        """ Get RunInformation """
        return self._runInfo

    def getFrameParams(self):
        """ Return AnalysisFrameParameters Structure """
        return self._frameParams

    # Public Interface

    def build(self):
        """ Build the Data Manager Instance """

        self.getRuntimeSettings().serialize()
             
        self._runInfo = CommonStructures.RunInformation( 
            self.getRuntimeSettings().getInputPaths(),
            self.getRuntimeSettings().getOutputPath() )

        self.initSampleShapeSizes()
        self.initBatchSizeData()
        self.initAnalysisFrameParams()

        return self

    def call(self):
        """ Run this Manager's Execution Method """
        
        return self

    def clean(self):
        """ Run Cleaning method on Instance """
        self._runInfo.serialize()
        super().clean()
        return self

    def addBatchData(self,batchData):
        """ Add Batch Data Instance to this Instance """
        self._batchDataObjs.append(batchData)
        self._runInfo.incrementExpectedNumSamples( batchData.getExpectedNumSamples() )
        self._runInfo.incrementActualNumSamples( batchData.getActualNumSamples() )
        return self

    # Private Interface

    def initSampleShapeSizes(self):
        """ Set the Sample Shape Sizes """
        shapeSampleA = self.getCollectionManager().getDesignMatrixA().getSampleShape()
        shapeSampleB = self.getCollectionManager().getDesignMatrixB().getSampleShape()
        
        # Copy Shapes Over
        for i in range(len(shapeSampleA)):
            self._runInfo.getShapeSampleA().append( shapeSampleA[i] )

        for j in range(len(shapeSampleB)):
            self._runInfo.getShapeSampleB().append( shapeSampleB[j] )

        # Return
        return self
            
    def initBatchSizeData(self):
        """ Inititalize Data related to batch Sizes """
        batchSizes = self.getSampleManager().getBatchSizes()
        for i in range(len(batchSizes)):
            self._runInfo.getBatchSizes().append( batchSizes[i] )
        return self

    def initAnalysisFrameParams(self):
        """ Initialize Analysis Frames Paramaters Structure """
        self._frameParams = Structural.AnalysisFramesParameters(
            samplesPerFrame=1024,
            samplesOverlap=768,
            headPad=1024,
            tailPad=2048,
            maxFrames=256)
        return self
