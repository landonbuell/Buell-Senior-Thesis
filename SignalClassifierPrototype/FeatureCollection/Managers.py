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
        

class SampleManager (Manager):
    """ SampleManager collects and organizes all data samples """

    def __init__(self):
        """ Constructor for SampleManager Instance """
        super().__init__()
        self._sampleDataBase    = np.array([],dtype=object)
        self._labelDictionary   = dict({})
        self._batchIndex        = 0

    def __del__(self):
        """ Destructor for SampleManager Instance """
        super().__del__()
        
    # Getters and Setters

    def getTarget(self,targetInt):
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

    def getNumClasses(self):
        """ Get the Number of Classes by entries in the Dictionary """
        return len(self._labelDictionary)

    def getBatchIndex(self):
        """ Get the Current Batch Index """
        return self._batchIndex

    def getBatchSize(self):
        """ Get the Size of Each Batch from App Settings """
        return Administrative.CollectionApplicationProtoype.AppInstance.getSettings().getBatchSize()

    # Public Interface

    def build(self):
        """ Gather + Organize all Audio Samples """
        super().build()

        self.readInputFiles()
        self.describe()

        return self

    def describe(self):
        """ Log description of state of this instance """

        # Basic Info
        messages = [
            "Number of Files Found: {0}".format(len(self)),
            "Entries in target label dictionary: {0}".format(self.getNumClasses())
            ]
        for msg in messages:
            # Log Each String as a Message
            self.logMessageInterface(msg)

        # Log the Label Dictionary
        for (key,val) in self._labelDictionary.items():
            msg = "{0:<32}\t{1:<16}\t{2:<32}".format(" ",key,val)
            self.logMessageInterface(msg,False)

        return self

    def createBatch(self,increment=True):
        """ Get an Array of Samples for the Next Batch """
        # Create the Batch Subset
        batchSize = self.getBatchSize()
        indexStart = self._batchIndex * batchSize
        batch = np.empty(shape=(batchSize,),dtype=object)
        
        # Populate Batch w/ Entries from Database
        for i in range(self.getBatchSize()):
            batch[i] = self._sampleDataBase[indexStart + i]

        if (increment == True):
            self._batchIndex += 1
        return batch

    # Private Interface

    def readInputFiles(self):
        """ Read Through All Input Files and Add to Sample Database """
        inputFiles = Administrative.CollectionApplicationProtoype.AppInstance.getSettings().getInputPaths()
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
            msg = "Found {0} samples".format(samplesInFile.shape[0])
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

    # Magic Methods

    def __item__(self,idx):
        """ Overload Index Operator """
        return self._sampleDataBase[idx];

    def __len__(self):
        """ Overload Length Operator """
        return self._sampleDataBase.shape[0]

    
class CollectionManager (Manager):
    """ CollectionManager organizes all Features Methods """

    def __init__(self):
        """ Constructor for CollectionManager Instance """
        super().__init__()

    def __del__(self):
        """ Destructor for CollectionManager Instance """
        super().__del__()


class MetadataManager (Manager):
    """ MetadataManager Aggregates all data from the Collection process """
    
    def __init__(self):
        """ Constructor for MetadataManager Instance """
        super().__init__()

    def __del__(self):
        """ Destructor for MetadataManager Instance """
        super().__del__()

