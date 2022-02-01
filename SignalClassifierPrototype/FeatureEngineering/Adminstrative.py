"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        FeatureEngineering
File:           Adminstrative.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys

import numpy as np

import CommonStructures

        #### CLASS DEFINITIONS ####

class PreprocessingQueue:
    """ Create A Pipeline of Preprocessing tools """

    def __init__(self,maxCapacity=16):
        """ Constructor for PreprocessingQueue Instance """
        self._data = np.zeros(shape=(maxCapacity,),dtype=object)

    def __del__(self):
        """ Destructor for PreprocessingQueue Instance """
        self._data = None

    # Getters and Setters

    def getSize(self):
        """ Get Number of elements in the Queue """
        size = 0
        for item in self._data:
            if (item != 0):
                size += 1
        return size

    # Public Interface 

    def enqueue(self,item):
        """ Enqueue and Item to the Pipeline """
        self._data = np.append(self._data,item)
        return self

    def dequeue(self,i=0):
        """ Remove an item from index i """
        self._data[i] = 0
        return self

    def fitAll(self,designMatrix):
        """ Fit each queue item w/ a design Matrix """
        for item in self._data:
            if (item == 0):
                continue
            item.fit(designMatrix)
        return self

    def transformAll(self,designMatrix):
        """ Transform design matrix w/ each queue item """
        for item in self._data:
            if (item == 0):
                continue
            designMatrix = item.transform(designMatrix)
        return designMatrix

    def processCollectionRun(self,runInfo,outputPath):
        """ Wrapper to Enable the Processing of a whole collection run """
        os.makedirs(outputPath,exist_ok=True)
        runInfoPath = os.path.join(outputPath,"runInformation.txt")
        runInfo.serialize(runInfoPath,batchLimit=-1)
        # Get all data = fit to preprocess
        allBatches = runInfo.loadAllSamples(True,False)
        matrixA = allBatches[0]

        # Fit all of the Data
        self.fitAll(matrixA)

        # Break into Batches
        batchesMatrixA = matrixA.splitIntoBatches(64)
        matrixA = None

        # Transform All + Export
        for batch in batchesMatrixA:
            self.transformAll(batch)
            batch.serialize()
        
        
        return self
    
    # Private Interface
    
    # Magic Methods

    def __len__(self):
        """ Return the Length of the Queue """
        return self.getSize()

    def __repr__(self):
        """ Return Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward Iterator through Queue """
        for item in self._data:
            if (item == 0):
                continue
            yield item



