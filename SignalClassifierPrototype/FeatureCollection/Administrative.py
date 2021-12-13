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

from numpy.core.numeric import outer

import Managers

        #### CLASS DEFINITIONS ####

class CollectionApplicationProtoype:
    """ 
    Contains All Application Functions for FeatureCollection
    """

    AppInstance = None

    def __init__(self,appSettings):
        """ Constructor for CollectionApplication Instance """
        CollectionApplicationProtoype.AppInstance = self

        self._settings          = appSettings 
        self._logger            = Logger()
        
        self._sampleManager     = None
        self._collectionManager = None
        self._rundataManager    = None

        
    def __del__(self):
        """ Destructor for CollectionApplication Instance """
        self.logDestruction()
        CollectionApplicationProtoype.AppInstance = None

    @staticmethod
    def constructApp(self,settings):
        """ Construct the Application """
        if (CollectionApplicationProtoype.AppInstance is None):
            CollectionApplicationProtoype.AppInstance = CollectionApplicationProtoype(settings)
        else:
            errMsg = "Can only have one instance of CollectionApplicationProtoype at runtime"
            raise RuntimeError(errMsg)
        return CollectionApplicationProtoype.AppInstance

    @staticmethod
    def destroyApp(self):
        """ Destroy the Aplication """
        if (CollectionApplicationProtoype.AppInstance is None):
            CollectionApplicationProtoype.AppInstance = CollectionApplicationProtoype(settings)
        else:
            errMsg = "Can only have one instance of CollectionApplicationProtoype at runtime"
            raise RuntimeError(errMsg)
        return CollectionApplicationProtoype.AppInstance
     
    # Getters and Setters

    @staticmethod
    def getAppInstance():
        """ Return the application Instance if it exists """
        if (CollectionApplicationProtoype.AppInstance is None):
            # App Does not Exist
            errMsg = "ERROR: CollectionApplicationProtoype has not been instantiated"
            raise RuntimeError(errMsg)
        else:
            return CollectionApplicationProtoype.AppInstance

    def getSettings(self):
        """ Return the Settings Instance """
        return self._settings

    def getSampleManager(self):
        """ Return the Sample Manager """
        return self._sampleManager

    def getCollectionManager(self):
        """ Return the Collection Manager """
        return self._collectionManager

    def getRundataManager(self):
        """ Return the Data Manager """
        return self._rundataManager()

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

    def startup(self):
        """ Run Application Startup Sequence """
        
        # Init the Managers
        self._sampleManager     = Managers.SampleManager()
        self._collectionManager = Managers.CollectionManager()
        self._rundataManager       = Managers.RundataManager()

        # Run Each Build Method
        self._sampleManager.build()
        self._collectionManager.build()
        self._rundataManager.build()

        return self

    def execute(self):
        """ Run Application Execution Sequence """
        
        for idx,size in enumerate(self._sampleManager.getBatchSizes()):

            # Run the Collection Manager on this Batch
            self._collectionManager.call(idx,size)

        self._rundataManager.call()
        return self

    def shutdown(self):
        """ Run Application Shutdown Sequence """

        return self
    
    # Internal Interface

    def logMessage(self,message,timeStamp=True):
        """ Log Message To User """
        self._logger.logMessage(message,timeStamp)
        return self

    @staticmethod
    def getDateTime() -> str:
        """ Get formatted DateTime as String """
        result = str(datetime.datetime.now())
        result = result.replace("-",".")
        result = result.replace(":",".")
        result = result.replace(" ",".")
        return result

    def logConstruction(self):
        """ Log Construction of Sample Manager """
        msg = "Constructing CollectionApplicationProtoype Instance ..."
        CollectionApplicationProtoype.AppInstance.logMessage(msg)
        return None

    def logDestruction(self):
        """ Log Construction of Sample Manager """
        msg = "Destroying CollectionApplicationProtoype Instance ..."
        CollectionApplicationProtoype.AppInstance.logMessage(msg)
        return None

    # Magic Methods

    def __repr__(self):
        """ Debugger representation of Instance """
        if (CollectionApplicationProtoype.AppInstance is None):
            # Not Yet Initialized
            return "No Instance"
        else:
            memAddress = str(hex(id(CollectionApplicationProtoype.AppInstance)))
            return "CollectionApplicationProtoype @ " + memAddress

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
        self._verbose       = 1
        self._logToConsole  = True
        self._logToFile     = False

        self.initInputPaths(pathsInput)
        self.initOutputPath(pathOutput)

    def __del__(self):
        """ Destructor for AppSettings Instance """
        pass

    # Getters and Setters

    def getStartupPath(self) -> str:
        """ Get Application Startup Path """
        return self._pathStartup

    def getInputPaths(self) -> set:
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

    def getLogToConsole(self) -> bool:
        """ Get T/F If Log to Console """
        return self._logToConsole

    def getLogToFile(self) -> bool:
        """ Get T/F IF Log to File """
        return self._logToFile

    # Public Interface

    def addInputPath(self,path) -> bool:
        """ Add New Input Path to the Set of Paths """
        fullPath = os.path.abspath(path)
        self._pathsInput.add(fullPath)
        return self

    def serialize(self)-> bool:
        """ Write the Settings Instance out to a text file """
        writer = AppSettingsSerializer(self,None)
        writer.call()
        return True

    @staticmethod
    def developmentSettingsInstance():
        """ Build an instance of runtime settings for development """
        result = AppSettings(
            pathsInput=["..\\lib\\DemoTargetData\\Y4.csv","..\\lib\\DemoTargetData\\Y3.csv"],
            pathOutput="..\\..\\..\\..\\audioFeatures\\outputTest_v1",
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

class AppSettingsSerializer:
    """ Class to Serialize AppSettings Instance """

    def __init__(self,settings,path=None):
        """ Constructor for AppSettingsSerializer Instance """
        self._settings  = settings
        self._path      = "-1"

        if (path is None):
            path = os.path.join(settings.getOutputPath(),"runtimeSettings.txt")
        else:
            path = path = os.path.join(path,"runtimeSettings.txt")

    def __del__(self):
        """ Destructor for AppSettingsSerializer Instance """
        self._settings = None

    def call(self):
        """ Serialize the Chosen Instance """
        path = os.path.join(self._settings.getOutputPath(),"runtimeSettings.txt")
        outline = lambda key,val : "{0:<32}\t{1:<128}\n".format(key,val)

        with open(path,"w") as outFileStream:
        
            # Write In/Out Paths
            outFileStream.write( outline("startupPath",self._settings.getStartupPath() ) )
            for i,val in enumerate( self._settings.getInputPaths() ):
                outFileStream.write( outline("inputPath_" + str(i),self._settings.getInputPaths()[i] ) )
            outFileStream.write( outline("outputPath",self._settings.getOutputPath() ) )

            # Write Collection Settings
            outFileStream.write( outline("BatchSize",self._settings.getBatchSize() ) )
            outFileStream.write( outline("ShuffleSeed",self._settings.getShuffleSelf() ) )
            outFileStream.write( outline("Verbose",self._settings.getVerbose() ) )

            # Write Log Levels
            outFileStream.write( outline("LogConsole",self._settings.getLogToConsole() ) )
            outFileStream.write( outline("LogFile",self._settings.getLogToFile() ) )

        return self
            

class Logger:
    """ 
    Handles all runtime Logging 
    """

    def __init__(self):
        """ Constructor for Logger Instance """      
        self._path          = None
        self._toConsole     = CollectionApplicationProtoype.AppInstance.getSettings().getLogToConsole()
        self._toFile        = CollectionApplicationProtoype.AppInstance.getSettings().getLogToFile()
        self.writeHeader()

    def __del__(self):
        """ Destructor for Logger Instance """
        self.writeFooter()

    # Public Interface

    def logMessage(self,message:str,timeStamp=True):
        """ Log Message to Console or Text """
        if (timeStamp == True):
            # Log Message w/ a TimeStamp
            self.logWithTimeStamp(message)
        else:
            # Log Message w/o a TimeStamp
            self.logWithoutTimeStamp(message)
        return self

    # Private Interface

    def logWithTimeStamp(self,msg:str):
        """ Log Message With TimeStamp """
        now = CollectionApplicationProtoype.getDateTime()
        if (self._toConsole):
            # Write Message to Console
            print("\t{0:<32}\t{1:<128}".format(now,msg))
        if (self._toFile):
            # Write Message to File
            errMsg = "ERROR: Log to File is not Implemented yet"
            raise NotImplementedError(errMsg)
        return self

    def logWithoutTimeStamp(self,msg:str):
        """ Log Message With TimeStamp """
        if (self._toConsole):
            # Write Message to Console
            print("\t{0:<128}".format(msg))
        if (self._toFile):
            # Write Message to File
            errMsg = "ERROR: Log to File is not Implemented yet"
            raise NotImplementedError(errMsg)
        return self

    def writeHeader(self):
        """ Write Header To Logger """
        header = [
            self.spacer(),
            "CollectionApplicationProtoype",
            CollectionApplicationProtoype.getDateTime(),
            self.spacer()
            ]
        # Log Each Line of the Header
        for msg in header:
            self.logWithoutTimeStamp(msg)
        return self

    def writeFooter(self):
        """ Write Footer To Logger """
        footer = [
            self.spacer(),
            "CollectionApplicationProtoype",
            CollectionApplicationProtoype.getDateTime(),
            self.spacer()
            ]
        # Log Each Line of the Header
        for msg in footer:
            self.logWithoutTimeStamp(msg)
        return self

    def spacer(self,numChars=64):
        """ Get a Spacer String """
        return "\n" + ("-" * numChars) + "\n"
    


        




