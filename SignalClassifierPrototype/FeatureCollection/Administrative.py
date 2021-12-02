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
import datetime

        #### CLASS DEFINITIONS ####

class CollectionApplication:
    """ 
    Contains All Application Functions for FeatureCollection
    """

    AppInstance = None

    def __new__(cls,appSettings):
        """ Allocator for CollectionApplication Instance """
        if (CollectionApplication.AppInstance is not None):
            # Instance Already Exists
            errMsg = "\tERROR: Can only have one instance of CollectionApplication at runtime"
            raise RuntimeError(errMsg)
        return

    def __init__(self,appSettings):
        """ Constructor for CollectionApplication Instance """
        self._settings          = appSettings 
        self._sampleManager     = None
        self._collectionManager = None
        self._exportManager     = None

        CollectionApplication.AppInstance   = self

    def __del__(self):
        """ Destructor for CollectionApplication Instance """
        CollectionApplication.AppInstance = None
        
    # Getters and Setters

    def getSettings(self):
        """ Return the Settings Instance """
        return self._settings

    def getCurrentDirectory(self):
        """ Return the Current Working Directory """
        return os.getcwd()

    def setCurrentDirectory(self,path):
        """ Set the Current Working Directory """
        if (os.path.isdir(path) == False):
            raise IsADirectoryError()
        os.chdir(path)
        return self

    # Public Interface

    def organizeAllSamples(self):
        """ Collect and Organize All Input Samples """
        pass

    def buildCollectionMethods(self):
        """ Construct the Queue of Collection Functions """
        pass

    def executeFeatureQueue(self):
        """ Run the Feature Queue on a Batch of Samples """
        pass

    def exportRemainingData(self):
        """ Export remaining data and teardown application """
        pass
    
    # Internal Interface

    def logMessage(self,message):
        """ Log Message To User """
            
        return self

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

class AppSettings:
    """
    Contains all runtime settings for duration of application
    """
    def __init__(self,pathsInput,pathOutput,batchSize=32,shuffleSeed=-1):
        """ Constructor for AppSettings Instance """
        self._pathStartup   = os.getcwd()
        self._pathsInput    = set()
        self._pathOutput    = None
        self._batchSize     = batchSize
        self._shuffleSeed   = shuffleSeed
        self._verbose       = True
        self._logToConsole  = True
        self._logToFile     = False

        self.initInputPaths(pathsInput)
        self.initOutputPath(pathOutput)

    def __del__(self):
        """ Destructor for AppSettings Instance """
        pass

    # Getters and Setters

    def getInputPaths(self) -> set[str]:
        """ Return List of Input Paths """
        return self._pathsInput

    def getOutputPath(self) -> str:
        """ Return Output Path """
        return self._pathOutput

    def getBatchSize(self) -> int:
        """ Return the Batch Size """
        return self._batchSize

    def getShuffleSelf(self) -> int:
        """ Return the Sufffle Seed """
        return self._shuffleSeed

    def getVerbose(self) -> bool:
        """ Return T/F if in Verbose Mode """
        return self._verbose

    def getLogToConsole(self)-> bool:
        """ Get T/F If Log to Console """
        return self._logToConsole

    def getLogToFile(self)-> bool:
        """ Get T/F IF Log to File """
        return self._logToFile

    # Public Interface

    def addInputPath(self,path) -> bool:
        """ Add New Input Path to the Set of Paths """
        fullPath = os.path.abspath(path)
        self._pathsInput.add(fullPath)

    def Serialize(self,path)-> bool:
        """ Write the Settings Instance out to a text file """
        return False

    @staticmethod
    def developmentSettingsInstance() -> AppSettings:
        """ Build an instance of runtime settings for development """
        result = AppSettings(
            pathsInput=["..\\lib\\DemoTargetData\\Y4.csv","..\\lib\\DemoTargetData\\Y3.csv"],
            pathOutput=".\\OutputTest_v0",
            batchSize=32,
            shuffleSeed=-1)
        return result

    # Private Interface

    def initInputPaths(self,pathSet):
        """ Initialize Set of Input Paths """
        for x in pathSet:
            self.addInputPath(x)
        return self

    def initOutputPath(self,output):
        """ Initialize the Output Path """
        fullOutput = os.path.abspath(output)
        if (os.path.isdir(fullOutput)):
            # Content may be overwritten
            msg = "WARNING: Output path exists. Contents may be over written"
        else:
            os.mkdir(fullOutput)
        self._pathOutput = fullOutput
        return self


class Logger:
    """ 
    Handles all runtime Logging 
    """

    Instance = None

    def __new__(cls,appSettings):
        """ Allocator for Logger Instance """
        if (Logger.Instance is not None):
            # Instance Already Exists
            errMsg = "\tERROR: Can only have one instance of Logger at runtime"
            raise RuntimeError(errMsg)
        return

    def __init__(self):
        """ Constructor for Logger Instance """
        self._path              = None
        Logger.Instance         = self

    def __del__(self):
        """ Destructor for Logger Instance """
        Logger.Instance = None

    # Public Interface

    def logMessage(self,message):
        """ Log Message to Console or Text """
        now = CollectionApplication.getDateTime()
        if (CollectionApplication.AppInstance.getSettings().getLogToConsole()):
            # Write Message to Console
            print("\t{0:<32}\t{1:<128}".format(now,message))
        if (CollectionApplication.AppInstance.getSettings().getLogToFile()):
            # Write Message to File
            errMsg = "ERROR: Log to File is not Implemented yet"
            raise NotImplementedError(errMsg)
        return self

    


        




