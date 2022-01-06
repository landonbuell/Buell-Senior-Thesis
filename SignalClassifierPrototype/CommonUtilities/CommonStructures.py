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
from typing import overload
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

    def listToString(self,inputList,delimiter=","):
        """ Convert Elements of list to string w/ delimiter """
        outputString = ""
        if len(inputList) == 0:
            # No Items in the Input List
            outputString += "-1,"
        else:
            # Items in the Input List
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
        """ Convert string to list of type """
        outputList = inputString.split(delimiter)
        if outType is not None:
            outputList = [outType(x) for x in outputList]
        return outputList

    def stringToIntList(self,inputString,delimiter):
        """ Convert string to list of type """
        outputList = []
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

    def getFeatures(self):
        """ Get Design Matrix as an Array """
        return self._data

    def setFeatures(self,x):
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
        return np.max(self._tgts)

    # public Interface

    def samplesInClass(self,classIndex):
        """ Create New Design Matrix of Samples that all belong to one class """
        if (classIndex not in self.getUniqueClasses()):
            # Not a Valid Class
            return DesignMatrix(1,self.getSampleShape())
        # Find where targets matches index
        mask = np.where(self._tgts == classIndex)[0]
        newTgts = self._tgts[mask]
        newData = self._data[mask]
        # Create the new Design Matrix, attach values + Return
        result = DesignMatrix(len(mask),self.getSampleShape())
        result.setLabels(newTgts)
        result.setFeatures(newData)
        return result


    def averageOfFeatures(self):
        """ Compute the Average of the Design Matrix Along each Feature """
        return np.mean(self._data,axis=0,dtype=np.float32)

    def varianceOfFeatures(self):
        """ Compute the Variance of the Design Matrix Along each Feature """
        return np.var(self._data,axis=0,dtype=np.float32)

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
    def deserialize(pathX,pathY,numSamples,shape):
        """ Read a design matrix from a file """
        reader = DesignMatrix.DesignMatrixDeserializer(
            pathX,pathY,numSamples,shape)
        matrix = reader.call()
        return matrix

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
            X = self._data.getFeatures()
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

        def __init__(self,pathX,pathY,numSamples,sampleShape):
            """ Constructor for DesignMatrixSerializer Instance """
            super().__init__("-1")
            self._pathX = pathX
            self._pathY = pathY
            self._data = DesignMatrix(numSamples,sampleShape)

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()

        def call(self):
            """ Run the Deserializer """
            self.validateInputPaths()
            self._data.setFeatures( self.readFeatures() )
            self._data.setLabels( self.readLabels() )
            return self._data

        # Private Interface

        def validateInputPaths(self):
            """ Check that Input Directories Exists """
            if (os.path.isfile(self._pathX) == False):
                # Path does not Exist
                FileNotFoundError(self._pathX)
            if (os.path.isfile(self._pathY) == False):
                # Path does not Exist
                FileNotFoundError(self._pathY)
            return True

        def readFeatures(self):
            """ Read the Feature Data from the File into the Design Matrix """
            shape = self._data.getShape()
            self._inFileStream = open(self._pathX,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.float32)         
            array = array.reshape( shape )         
            return array

        def readLabels(self):
            """ Read the Feature Data from the File into the Design Matrix """
            self._inFileStream = open(self._pathY,"rb")
            fileContents = self._inFileStream.read()
            self._inFileStream.close()
            array = np.frombuffer(fileContents,dtype=np.int16)               
            return array
 
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

class ClassOccuranceData:
    """ Store and Export Inforamtion on the Occurance of Each Class """

    def __init__(self):
        """ Constructor for ClassOccuranceData Instance """
        self._labelNames = dict({})
        self._labelOccurances = dict({})

    def __del__(self):
        """ Destructor for ClassOccuranceData Instance """
        pass

    # Getters and Setters

    def getNameFromInt(self,labelInt):
        """ Get the Class Name From an Integer """
        try:
            return self._labelNames[labelInt]
        except KeyError:
            return "unknown"

    def getIntFromName(self,labelStr):
        """ Get the Class Index from a Name """
        for (key,val) in self._labelNames.items():
            # Each Key-Val Pair
            if (val == labelStr):
                return key
        # Did not Find Int
        return -1

    def getUniqueClassInts(self):
        """ Get All Unique Classes as Ints """
        return self._labelNames.keys()

    def getUniqueClassStrs(self):
        """ Get All Unique Classes as Strs """
        return self._labelNames.values()

    # Public Interface

    def update(self,targetInt,targetStr=None):
        """ Update the Internal Dictionarys """
        if (targetInt not in self._labelNames.keys()):
            # Not in Dictionary
            if (targetStr is not None):
                # Target String Label is given
                self._labelNames.update({targetInt:targetStr})
                self._labelOccurances.update({targetInt:0})
            else:
                # Target String Label is not given
                self._labelNames.update({targetInt:"NONE"})
                self._labelOccurances.update({targetInt:0})
      
        # Now Update the Counter
        self._labelOccurances[targetInt] += 1
        return self


    def serialize(self,path):
        """ Serialize this Instance to Local Path """
        writer = ClassOccuranceData.ClassOccuranceDataSerializer(self,path)
        writer.call()
        return self
    
    @staticmethod
    def deserialize(path):
        """ Deserialize an Instance from local path """
        return self

    # Private Interface

    class ClassOccuranceDataSerializer(Serializer):
        """ Class to Serialize Occurance Data """
        
        def __init__(self,data,path):
            """ Constructor for ClassOccuranceDataSerializer Instance """
            super().__init__(data,path)

        def __del__(self):
            """ Destructor for ClassOccuranceDataSerializer Instance """
            super().__del__()

        def call(self):
            """ Serialize the Instance """
            self._outFileStream = open(self._outputPath,"w")
            self.writeHeader()
            header = "\t{0:<16}{1:<32}{2:<16}\n".format("Int","Name","Count")
            self._outFileStream.write( header )

            # Write Body
            for row in self._data:
                msg = "{0:<16}{1:<32}{2:<16}\n".format(row[0],row[1],row[2])
                self._outFileStream.write( msg )

            # Close + Exit
            self.writeFooter()
            self._outFileStream.close()
            return self

    # Magic Methods:

    def __iter__(self):
        """ Forward Iterator over samples """
        for labelInt in self.getUniqueClassInts():
            labelStr = self._labelNames[labelInt]
            labelCnt = self._labelOccurances[labelInt]
            yield (labelInt,labelStr,labelCnt)

class RunInformation:
    """
    Class to Hold and Use all Metadata related to a feature collection Run
    """

    def __init__(self,inputPaths,outputPath,
                 numSamplesExpected=0,numSamplesRead=0):
        """ Constructor for RunInformation Instance """
        self._pathsInput        = inputPaths
        self._pathOutput        = outputPath

        self._numSamplesExpt    = 0
        self._numSamplesRead    = 0

        self._shapeSampleA      = []
        self._shapeSampleB      = []

        self._featureNamesA     = []
        self._featureNamesB     = []

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

    def setExpectedNumSamples(self,num):
        """ Set the number of samples expected to process """
        self._numSamplesExpt = num
        return self

    def getActualNumSamples(self):
        """ Get the number of samples actually processed """
        return self._numSamplesRead

    def setActualNumSamples(self,num):
        """ Set the number of samples actually processed """
        self._numSamplesRead = num
        return self

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

    def setShapeSampleA(self,shape):
        """ Set the Shape of samples in design matrix A """
        self._shapeSampleA = shape
        return self

    def getShapeSampleB(self):
        """ Get the shape of samples in design matrix B """
        return self._shapeSampleB

    def setShapeSampleB(self,shape):
        """ Set the Shape of samples in design matrix B """
        self._shapeSampleB = shape
        return self

    def getFeatureNamesA(self):
        """ Get the List of Feature names for Matrix A """
        return self._featureNamesA

    def setFeatureNamesA(self,names):
        """ Set the List of Feature names for Matrix A """
        self._featureNamesA = names
        return self

    def getFeatureNamesB(self):
        """ Get the List of Feature names for Matrix B """
        return self._featureNamesB

    def setFeatureNamesB(self,names):
        """ Set the List of Feature names for Matrix B """
        self._featureNamesB = names
        return self

    def getBatchSizes(self):
        """ Get the Sizes of all Batches """
        return self._batchSizes

    def setBatchSizes(self,sizes):
        """ Set the Sizes of all batches """
        self._batchSizes = sizes
        return self

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

    def loadAllSamples(self,maxSamples=65536):
        """ Load All Samples From All batches """
        sampleIndex = 0
        matrixA = DesignMatrix(self._numSamplesRead,self._shapeSampleA)
        matrixB = DesignMatrix(self._numSamplesRead,self._shapeSampleB)
        
        # Iterate through Each Batch
        for batchIndex,numSamples in enumerate(self._batchSizes):
            batchMatricies = self.loadBatch(batchIndex)
            # Copy Into parent Matrices
            for sample in range(numSamples):
                matrixA._data[sampleIndex] = batchMatricies[0]._data[sample]
                matrixB._data[sampleIndex] = batchMatricies[1]._data[sample]
                sampleIndex += 1
        # Loaded All Batches - Return Total Design Matrices
        return (matrixA,matrixB,)



    def loadBatch(self,index):
        """ Load In All Data from a chosen batch Index """
        numSamples = self._batchSizes[index]
        # Set the Matrix Paths
        name = lambda idx,descp : "batch" + str(idx) + "_" + str(descp) + ".bin"
        pathXa  = os.path.join(self._pathOutput, name(index,"Xa") )
        pathXb  = os.path.join(self._pathOutput, name(index,"Xb") )
        pathY   = os.path.join(self._pathOutput, name(index,"Y") )

        # Load in the matrices
        matrixA = DesignMatrix.deserialize(pathXa,pathY,numSamples,self.getShapeSampleA() )
        matrixB = DesignMatrix.deserialize(pathXb,pathY,numSamples,self.getShapeSampleB() )
        return (matrixA,matrixB,)

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
        runInfoPath = os.path.join(path,"runInformation.txt")
        if (os.path.isfile(runInfoPath) == False):
            # RunInfo File
            errMsg = "ERROR: run information file not found at '{0}' ".format(runInfoPath)
            FileNotFoundError(errMsg)
        reader = RunInformation.RunInformationDeserializer(runInfoPath)
        runInfo = reader.call()
        return runInfo


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

            # Write Feature Name Details
            featureNamesA = self.listToString(self._data.getFeatureNamesA(),",")
            featureNamesB = self.listToString(self._data.getFeatureNamesB(),",")
            self._outFileStream.write( self._outFmtStr("FeatureNamesA", featureNamesA))
            self._outFileStream.write( self._outFmtStr("FeatureNamesB", featureNamesB))

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
            self._inFileContents = None

        def __del__(self):
            """ Destructor for DesignMatrixSerializer Instance """
            super().__del__()
            self._inFileContents = None

        def call(self):
            """ Serialize the RunInfo Instance """    
            self._inFileStream = open(self._inputPath,"r")
            self._inFileContents = self._inFileStream.readlines()
            
            # Find all of the Necessary parts
            runInfo = self.parseAllFeilds()
            return runInfo

        # Private Interface

        def parseAllFeilds(self):
            """ Find all of the Feilds to Create the RunInformation Instance """
            
            # Parse the feilds from the RunInfo File
            pathsInput      = self.findAndParseStrs("InputPath")
            pathOutput      = self.findAndParseStrs("OutputPath")[-1]
            samplesExpected = self.findAndParseInts("ExpectedSamples")[-1]
            samplesActual   = self.findAndParseInts("ProcessedSamples")[-1]
            shapeSampleA    = self.findAndParseInts("ShapeSampleA")
            shapeSampleB    = self.findAndParseInts("ShapeSampleB")
            batchSizes      = self.findAndParseInts("BatchSizes")
            
            # Assign the Feilds to the instance
            runInfo = RunInformation(pathsInput,pathOutput)
            runInfo.setExpectedNumSamples(samplesExpected)
            runInfo.setActualNumSamples(samplesActual)
            runInfo.setShapeSampleA(shapeSampleA)
            runInfo.setShapeSampleB(shapeSampleB)
            runInfo.setBatchSizes(batchSizes)
            return runInfo

        def findAndParseStrs(self,keyword,):
            """ Find All words with token and return as list of Strings"""
            result = []
            for line in self._inFileContents:
                tokens = line.split()
                if tokens[0].startswith(keyword):
                    result.append(tokens[-1].strip())
            return result

        def findAndParseInts(self,keyword):
            """ Find All words with token and return as list of Strings"""
            result = self.findAndParseStrs(keyword)
            result = result[0].split(',')
            result = ["".join(ch for ch in x if ch.isdigit()) for x in result]
            result = [int(x) for x in result if x != '']
            return result
        

    # Magic Methods

    def __repr__(self):
        """ Debug Representation of Instance """
        return str(self.__class___) + " @ " + str(hex(id(self)))