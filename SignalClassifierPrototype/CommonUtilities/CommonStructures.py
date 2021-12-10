"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        CommonUtilities
File:           Structures.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import numpy as np


        #### CLASS DEFINITIONS ####

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

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

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
        writer = DesignMatrix.DesignMatrixSerializer(path,self)
        writer.call()
        return self

    @staticmethod
    def deserialize(self,path):
        """ Read this design matrix from a file """
        return self

    def clearData(self):
        """ Clear All Entries in this Array """
        self._data = np.zeros(shape=self.getShape(),dtype=np.float32)
        self._tgts = np.zeros(shape=self.getNumSamples(),dtype=np.int16)
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

    

    # Magic Methods 

    def __str__(self):
        """ String Representation of Instance """
        return str(self.__class__) + " w/ shape: " + str(self.getShape())

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

    def __iter__(self):
        """ Forward-Iterator through Design Matrix """
        for i in range(self._data.shape[0]):
            yield self._data[i]

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

class DesignMatrixSerializer:
    """ Class to Serialize a DesignMatrixInstance """
        
    def __init__(self,outputPath,matrix):
        """ Constructor for DesignMatrixSerializer Instance """
        self._outputPath    = outputPath
        self._matrix        = matrix
        self._fileHandle     = None

    def __del__(self):
        """ Destructor for DesignMatrixSerializer Instance """
        self._matrix = None
        if (self._fileHandle is not None):
            self._fileHandle.close()

    def call(self):
        """ Run the Serializer """
        numSamples = self._matrix.getNumSamples()
        X = self._matrix.getData()
        Y = self._matrix.getLabels()
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

class DesignMatrixDeserializer:
    """ Class to Serialize a DesignMatrix Instance """

    def __init__(self,localPath):
        """ Constructor for DesignMatrixSerializer Instance """
        self._outputPath    = localPath
        self._matrix        =
        self._fileHandle    = None

    def __del__(self):
        """ Destructor for DesignMatrixSerializer Instance """
        self._matrix = None
        if (self._fileHandle is not None):
            self._fileHandle.close()

    def call(self):
        """ Run the Serializer """

        return self