"""
Repository:     Buell-Senior-Thesis
Solution:       SignalClassifierPrototype
Project:        CommonUtilities
File:           Structures.py
 
Author:         Landon Buell
Date:           December 2021
"""

        #### IMPORTS ####

import os
import sys
import numpy as np

        #### CLASS DEFINITIONS ####

class Serializer:
    """ Abstract Base Class for all Serializer Classes """

    def __init__(self,data,path):
        """ Constructor for Serializer Abstract Class """
        self._data              = data
        self._outputPath        = path
        self._outFileStream     = None
        self._outFmtStr         = lambda key,val :  "{0:<32}\t{1:<128}\n".format(key,val)

    def __del__(self):
        """ Destructor for Serializer Abstract Class """
        if (self._outFileStream is not None):
            self._outFileStream.close()
        return

    def call(self):
        """ Write Object to OutputStream """

        return False

    def listToString(self,inputList,delimiter=" "):
        """ Convert Elements of list to string w/ delimiter """
        outputString = ""
        for item in inputList:
            outputString += str(item) + delimiter
        return outputString.strip()

    def writeHeader(self):
        """ Add Header To Output """
        self._outFileStream.write(self.__repr__() + "\n")
        self._outFileStream.write("-"*64 + "\n")
        return self

    def writeFooter(self):
        """ Add Header To Output """
        self._outFileStream.write("-"*64 + "\n")
        self._outFileStream.write(self.__repr__() + "\n")
        return self

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

class Deserializer:
    """ Abstract Base Class for all Deserializer Classes """

    def __init__(self,path):
        """ Constructor for Deserializer Abstract Class """
        self._data              = None
        self._inputPath         = path
        self._inFileStream      = None

    def __del__(self):
        """ Destructor for Deserializer Abstract Class """
        self._data = None
        if (self._inFileStream is not None):
            self._inFileStream.close()
        return

    def call(self):
        """ Read Object From inputStream """

        return False

    def stringToList(self,inputString,delimiter=" ",outType=None):
        """ Convert Elements of list to string w/ delimiter """
        outputList = inputString.split(delimiter)
        outputList = [outType(x) for x in outputList]
        return outputList

    def __repr__(self):
        """ Debugger Representation of Instance """
        return str(self.__class__) + " @ " + str(hex(id(self)))

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

    def getShape(self):
        """ Get Total Shape of Design Matrix """
        shape = [self._numSamples]
        for axisShape in self._sampleShape:
            shape.append(axisShape)
        return tuple(shape)

    def getSampleShape(self):
        """ Get the Shape of Each Sample in the Design Matrix """
        return self._sampleShape

    def getNumFeatures(self):
        """ Get the Total Number of Features for each sample """
        numFeatures = 1
        for axisSize in self._sampleShape:
            numFeatures *= axisSize
        return numFeatures

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
        self._sampleShape = tuple(x.shape[1:])
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

    def serialize(self,pathX=None,pathY=None):
        """ Write this design matrix out to a file """   
        writer = DesignMatrix.DesignMatrixSerializer(self,pathX,pathY)
        success = True
        try:          
            writer.call()
        except Exception as err:
            print("\t\tDesignMatrix.serialize()" + err)
            success = False
        return success

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

    class DesignMatrixSerializer(Serializer):
        """ Class to Serialize a DesignMatrixInstance """
        
        def __init__(self,matrix,pathX=None,pathY=None):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__(matrix,None)
            self._pathX =   pathX
            self._pathY =   pathY
            
            
        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Run the Serializer """
            self.validateOutputs()
            if (self._pathX is not None):
                self.writeDataX()
            if (self._pathY is not None):
                self.writeDataY()
            return self

        def writeDataX(self):
            """ Write the Design Matrix Data """
            numSamples = self._data.getNumSamples()
            X = self._data.getData()
            self._outFileStream = open(self._pathX,"wb")
            for i in range(numSamples):
                row = X[i].flatten().tobytes()
                self._outFileStream.write( row )
            # Close + Return
            self._outFileStream.close()
            return self

        def writeDataY(self):
            """ Write the Design Matrix Labels """
            numSamples = self._data.getNumSamples()
            Y = self._data.getLabels()
            self._outFileStream = open(self._pathY,"wb")
            for i in range(numSamples):
                row = Y[i].flatten().tobytes()
                self._outFileStream.write( row )
            # Close + Return
            self._outFileStream.close()
            return self

        def validateOutputs(self):
            """ Validate that Both Output Paths Make Sense """
            if (self._pathX is None and self._pathY is None):
                # Both Cannot be none - Nothing will be written
                errMsg = "Both X and Y export paths cannot be None"
                raise ValueError(errMsg)
            elif (self._pathX == self._pathY):
                # Both cannot be the same - will overwrite each other
                errMsg = "X and Y paths cannot be indentical"
                raise ValueError(errMsg)
            else:
                return self

    class DesignMatrixDeserializer(Deserializer):
        """ Class to Serialize a DesignMatrix Instance """

        def __init__(self,path,numSamples,sampleShape):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__(path)
            self._data = DesignMatrix(numSamples,sampleShape)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Run the Deserializer """

            return False
    
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



class RunInformation:
    """
    Class to Hold and Use all Metadata related to a feature collection Run
    """

    def __init__(self,inputPaths,outputPath):
        """ Constructor for RunInformation Instance """
        self._pathsInput        = inputPaths
        self._pathOutput        = outputPath

        self._numSamplesExpt    = 0
        self._numSamplesRead    = 0

        self._shapeSampleA      = []
        self._shapeSampleB      = []

        self._batchSizes        = []


    def __del__(self):
        """ Destructor for RunInformation Instance """
        pass

    # Getters and Setters

    def getRunInfoPath(self):
        """ Get the Path to the RunInfo Metadata """
        return os.path.join(self._pathOutput,"runInformation.txt")

    def getInputPaths(self) -> set:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getExpectedNumSamples(self):
        """ Get the number of samples expected to process """
        return self._numSamplesExpt

    def getActualNumSamples(self):
        """ Get the number of samples actually processed """
        return self._numSamplesRead

    def incrementExpectedNumSamples(self,amount):
        """ Increment the Expected number of Samples by the amount """
        self._numSamplesExpt += amount
        return self

    def incrementActualNumSamples(self,amount):
        """ Increment the Actual number of Samples by the amount """
        self._numSamplesRead += amount
        return self

    def getShapeSampleA(self):
        """ Get the shape of samples in design matrix A """
        return self._shapeSampleA

    def getShapeSampleB(self):
        """ Get the shape of samples in design matrix B """
        return self._shapeSampleB

    def getBatchSizes(self):
        """ Get the Sizes of all Batches """
        return self._batchSizes

    def getNumBatches(self):
        """ Get the Number of Batches in the run """
        return len(self._batchSizes)

    def getSizeSampleA(self):
        """ Get the size of each sample in design matrix A """
        size = 1
        for x in self._shapeSampleA:
            size *= x
        return size

    def getSizeSampleB(self):
        """ Get the size of each sample in design matrix B """
        size = 1
        for x in self._shapeSampleB:
            size *= x
        return size
    
    # Public Interface 

    def serialize(self,path=None):
        """ Serialize this Instance to specified Path """
        if (path is None):
            path = self.getRunInfoPath()
        writer = RunInformation.RunInformationSerializer(self,path)
        success = True
        try:
            writer.call()
        except Exception as err:
            print("\t\tRunInformation.serialize()" + err)
            success = False
        return success


    @staticmethod
    def deserialize(path):
        """ Deserialize this instance from specified path """
        return False


    # Private Interface

    class RunInformationSerializer(Serializer):
        """ Class to Serialize Run Information to a Local Path """

        def __init__(self,runInfo,path):
            """ Constructor for RunInformationSerializer Instance """
            super().__init__(runInfo,path)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Serialize the RunInfo Instance """          

            self._outFileStream = open(self._outputPath,"w")
            self.writeHeader()

            # Write Paths
            for i,path in enumerate(self._data.getInputPaths()):
                self._outFileStream.write( self._outFmtStr("InputPath_" + str(i),path ) )
            self._outFileStream.write( self._outFmtStr("OutputPath",self._data.getOutputPath() ) )

            # Write Sample Details
            self._outFileStream.write( self._outFmtStr("ExpectedSamples",self._data.getExpectedNumSamples() ) )
            self._outFileStream.write( self._outFmtStr("ProcessedSamples",self._data.getActualNumSamples() ) )

            # Write Sample Shape Detials
            shapeSampleA = self.listToString(self._data.getShapeSampleA(),",")
            shapeSampleB = self.listToString(self._data.getShapeSampleB(),",")
            self._outFileStream.write( self._outFmtStr("ShapeSampleA",shapeSampleA ) )
            self._outFileStream.write( self._outFmtStr("ShapeSampleB",shapeSampleB ) )

            # Write Batch Details
            batchSizes = self.listToString(self._data.getBatchSizes(),",")
            self._outFileStream.write( self._outFmtStr("BatchSizes",batchSizes ) )

            # Close + Return
            self.writeFooter()
            self._outFileStream.close()
            return True

    class RunInformationDeserializer(Deserializer):
        """ Class to Deserialize Run Information from a Local Path """

        def __init__(self,path):
            """ Constructor for RunInformationSerializer Instance """
            super().__init__(path)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Serialize the RunInfo Instance """          
            return True

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class___) + " @ " + str(hex(id(self)))