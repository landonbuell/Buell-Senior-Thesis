"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import datetime
import argparse

import scipy.io.wavfile as sciowav

import PlottingUtilities as plot_utils

"""
SystemUtilities.py - "SystemUtilities"
    Contains Variables, Classes, and Definitions for Lower program functions
    Backends, Data structure objects, os & directory organization and validations

"""

            #### DATA STRUCTURE CLASSES ####

class FileObject:
    """
    Create File 
    --------------------------------
    datarow (arr) : array of shape (1 x 3):
        has form:
        | Fullpath  | Target Int    | Target Str |
    --------------------------------
    Return instatiated file_object class
    """

    def __init__(self,datarow):
        """ Initialize Object Instance """
        # Directory & File Meta Data
        self.fullpath = datarow[0]              # set full file path
        dir_tree = self.fullpath.split('\\')     
        self.filename = dir_tree[-1]            # filename
        self.extension = "."+self.filename.split(".")[-1]

        # Set Target Int/Str for this File
        try:
            self.targetInt = int(datarow[1])    # target as int             
        except:
            self.targetInt = None    # unknown int
        try:
            self.targetStr = str(datarow[2])    # target as str
        except:
            self.targetStr = None    # unknown str
        
    def ReadFileWAV (self):
        """ Read raw .wav file data from local path """
        rate,data = sciowav.read(self.fullpath)    
        data = data.reshape(1,-1).ravel()   # flatten waveform
        self.rate = rate                # set sample rate
        self.waveform = data/np.abs(np.amax(data))
        self.n_samples = len(self.waveform)
        return self                     # return self

    def ReadFileTXT (self):
        """ Read raw .txt file data from local path """
        data = np.loadtxt(self.fullpath,dtype=float,usecols=0,unpack=True)
        self.waveform = data.flatten()
        assert np.ndim(self.waveform) == 1
        self.rate = 44100
        return self

class OutputStructure :
    """
    Data Structure to Store and Output History or Prediction arrays
    --------------------------------
    programMode (str) : Current Program Execution Model
    exportPath (str) : Local path where structure is exported to
    --------------------------------
    Return Instantiated output Structure Class
    """

    def __init__(self,programMode,exportPath):
        """ Initialize Class Object instance """
        if programMode == "Train":      # output for training history
            self.cols = ["Loss Score","Accuracy","Precision","Recall"]
        elif programMode == "Predict":  # output for predictions
            self.cols = ["Filepath","Int Label","Str Label",
                         "Int Prediction","Str Prediction","Confidence"]
        else:
            print("\n\tERROR! - File mode not recognized!")
            self.cols = []
            raise BaseException()
        self.exportPath = exportPath
        self.InitData()
    
    def InitData (self):
        """ Initialize Output Structure """
        self.Data = pd.DataFrame(data=None,columns=self.cols)       # create frame
        self.Data.to_csv(path_or_buf=self.exportPath,index=False)   # export frame
        return self

    def UpdateData (self,X):
        """ Update Data in Output Frame """
        frame = pd.DataFrame(data=X)        # create new dataframe
        frame.to_csv(path_or_buf=self.exportPath,columns=self.cols,
                     index=False,header=False,mode='a') # append the output frame
        return self

class CategoryDictionary :
    """
    Category Dictionary Maps an integer class to string class
        and vice-versa
    """

    def __init__(self,localPath,modelName):
        """ Intitialize CategoryDictionary Instance """
        self.localPath = localPath
        self.modelName = modelName
        self.fileName = modelName+"Categories.csv"
        self.filePath = os.path.join(self.localPath,self.fileName)
        self.nClasses = None

    def __Call__(self,newModel,fileObjs):
        """ Run Category Encode / Decode Dictionary """
        if newModel == True:        
            # A new Model is created, Overwrite existing File
            print("\tBuilding new Encode/Decode Dictionary...")
            self.encoder,self.decoder = self.BuildCategories(fileObjs)    
            self.ExportCategories()
        else:
            # Check if the encoder / Decoder Exists:
            if os.path.isfile(self.filePath) == True:
                print("\tFound Encode/Decode Dictionary")
                # the file exists, load it as enc/dec dict
                self.encoder,self.decoder = self.LoadCategories()
                self.nClasses = max(self.encoder.values()) + 1  # get number of classes
            else:
                # File does not exists, make a new Dictionary
                print("\tCould not find Encode/Decode Dictionary, building new")
                self.encoder,self.decoder = self.BuildCategories(fileObjs)    
                self.ExportCategories()
        return self

    def LoadCategories (self):
        """ Load File to Match Int -> Str Class """
        decoder = {}
        encoder = {}
        rawData = pd.read_csv(self.filePath)
        Ints,Strs = rawData.iloc[:,0],rawData.iloc[:,1]
        cnts = rawData,iloc[:,2]
        self.nClasses = len(cnts.to_numpy())
        for Int,Str in zip(Ints,Strs):      # iterate by each
            encoder.update({str(Str):int(Int)})
            decoder.update({int(Int):str(Str)})
        return encoder,decoder

    def ExportCategories (self):
        """ Export Decode Dictionary (Int -> Str) """
        decodeList = sorted(self.decoder.items())
        cols = ["Target Int","Target Str"]
        decodeFrame = pd.DataFrame(data=decodeList,columns=cols)
        decodeFrame["Counts"] = self.classCounter
        decodeFrame.to_csv(self.filePath,index=False)
        return self

    def BuildCategories (self,fileObjs):
        """ Construct Dictionary to map STR -> INT """
        encoder = {}            # empty enc dict
        decoder = {}            # empty dec dict
        targetInts = [x.targetInt for x in fileObjs]
        targetStrs = [x.targetStr for x in fileObjs]
       
        self.nClasses = np.max(targetInts) + 1
        self.classCounter = np.zeros(shape=(self.nClasses),dtype=int)

        for x,y in zip(targetInts,targetStrs):
            if x not in encoder.keys():     
                decoder.update({x:y})   # add int : str pair
            self.classCounter[x] += 1   # add to cntr

        # Organize in numerical order
        sortedItems = sorted(decoder.items())
        encoder = {}      # reset encoder
        for (targetInt,targetStr) in sortedItems:
            encoder.update({targetStr:targetInt})
            decoder.update({targetInt:targetStr})
        return encoder,decoder

            #### PROGRAM PROCESSING CLASS ####

class ArgumentValidator :
    """ Process and Validate Command Line Arguments """

    def __init__(self):
        """ Initialize ArgumentValidator Instance """
        self.MakeArgumentParser()

        # Parse & return Args
        if self.GetParsedArguments() == False:
            # Can't Get CL-Args, use hardcoded ones
            self.HardCodedVars()

        # make Sure Args are in acceptable values
        assert self.modelName not in [None," "]
        assert self.programMode in ['train','train-predict','predict']
        assert self.newModel in [True,False]

        self.ValidateLocalPaths(self.GetLocalPaths)   
        if (self.programMode == 'predict') and (self.newModel == True):
            print("\n\tERROR! -  Cannot run predictions on an untrained (new) Model!")
            raise BaseException()

    def MakeArgumentParser(self):
        """ Construct Argument Parser Object """
        self.argumentParser = argparse.ArgumentParser(description="Run Signal Classifier Program")
        
        self.argumentParser.add_argument("dataPath",type=str,
                                         help="local path where target data is held")
        self.argumentParser.add_argument("exportPath",type=str,
                                         help="local path where model/dictionary data is/will be stored")
        self.argumentParser.add_argument("modelPath",type=str,
                                         help="local path where model history/predictions are exported")
      
        self.argumentParser.add_argument("programMode",type=str,
                                         help="Mode of Program Execution: ['train','train-predict','predict']")
        self.argumentParser.add_argument("modelName",type=str,
                                         help="Name assigned to model for human-indentification")
        self.argumentParser.add_argument("newModel",type=str,
                                         help="If True, existing model of same name is overwritten")
        return self

    def GetParsedArguments(self):
        """ Collect All Command-Line Parsed Arguments in a List """
        try:
            arguments = self.argumentParser.parse_args()
            self.readPath = arguments.dataPath
            self.exportPath = arguments.exportPath
            self.modelPath = arguments.modelPath
            self.programMode = arguments.programMode
            self.modelName = arguments.modelName
            self.newModel = self.StringToBoolean(arguments.newModel)
            return True
        except:
            return False

    def HardCodedVars(self):
        """ These are Hard-coded vars, used only for development """
        parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier'
        self.readPath = os.path.join(parent,'Target-Data')
        #self.readPath = os.path.join(parent,'ChaoticSynth-Data')
        self.modelPath = os.path.join(parent,'Model-Data')
        self.exportPath = os.path.join(parent,'Output-Data')
        self.programMode = "train-predict"
        self.modelName = "ChaoticSynthClassifier"
        self.newModel = True

    @property
    def GetLocalPaths (self):
        """ Return Necessary Directory Paths """
        return (self.readPath,self.exportPath,self.modelPath)

    @property
    def GetSystemParams (self):
        """ Return Important Parameters for Creating Models """
        return (self.programMode,self.modelName,self.newModel)

    @staticmethod
    def StringToBoolean(value):
        """ Convert String entry to boolen value """
        value = str(value).lower()      # conv to lowercase str
        if value in ["true","t","yes","y","1"]:
            return True
        elif value in ["false","f","no","n","0"]:
            return False
        else:
            raise ValueError("\n\t Unrecognized value!")

    def ValidateLocalPaths (self,paths=[]):
        """ Confirm Existance of All Directories in List """
        for path in paths:
            if os.path.isdir(path) == False:
                # Directory does not exist
                raise NotADirectoryError(path)
        return self
        
class ProgramInitializer:
    """
    Object to handle all program preprocessing
    --------------------------------
    pathList (list) : List of 3 Local Directory paths:
        readPath - local path where target data is held       
        exportPath - local path where training history / predictions are exported
        modelPath - local path where model/dictionary data is/will be stored
    argsList (list): List of 3 Import Arguments for Program execution:
        mode (str) : String indicating which mode to execute program with
        modelname (str) : Name for Neural Network
        newModel (bool): If True, create new Neural Network Models
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,pathList=[None,None,None],argsList=[None,None,None]):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        starttime = dt_obj.isoformat(sep='.',timespec='auto')
        self.starttime = starttime.replace(':','.').replace('-','.')
        print("Time Stamp:",self.starttime)

        # Establish Paths
        self.readPath = pathList[0]
        self.exportPath = pathList[1]
        self.modelPath =  pathList[2]
        
        # Establish Params
        self.programMode = argsList[0]      
        self.modelName = argsList[1]
        self.newModel = argsList[2]
        
    def __repr__(self):
       """ Return String representation of Object/Instance """
       return "ProgramInitializer performs preprocessing for program parameters "

    def __Call__(self):
        """ Run Program Start Up Processes """     
        print("\nRunning Main Program.....\n")
        self.files = self.CollectFiles()        # find CSV files
        fileObjects = self.CreateFileobjs()     # file all files
        self.n_files = len(fileObjects)         # get number of files
        
        # Construct Encoder / Decoder Dictionaries
        self.categories = CategoryDictionary(self.modelPath,self.modelName)
        self.categories.__Call__(self.newModel,fileObjects)
        self.n_classes = self.categories.nClasses

        # Final Bits
        self.StartupMesseges           # Messages to User
        fileObjects = np.random.permutation(fileObjects)    # permute
        return fileObjects

    @property
    def GetLocalPaths (self):
        """ Return Necessaru Directory Paths """
        return (self.readPath,self.exportPath,self.modelPath)

    @property
    def GetModelParams (self):
        """ Return Important Parameters for Creating Models """
        return (self.modelName,self.newModel,self.n_classes,self.starttime)

    @property
    def GetDecoder (self):
        """ Return Decoder Dictionary, maps Int -> Str """
        return self.categories.decoder
            
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("\tCollecting data from:\n\t\t",self.readPath)
        print("\tStoring/Loading models from:\n\t\t",self.modelPath)
        print("\tExporting Predictions to:\n\t\t",self.exportPath)
        print("\tModel name:",self.modelName)
        print("\tCreating new models?",self.newModel)
        print("\tNumber of Files found:",self.n_files)
        print("\tNumber of Categories found:",self.n_classes)
        print("\n")
        return None

    def CollectFiles (self,exts='.csv'):
        """ Search Local Directory for all files matching given extension """
        csvFiles = []
        for file in os.listdir(self.readPath):  # in the path
            if file.endswith(exts):     # is proper file type
                csvFiles.append(file)   # add to list
        return csvFiles                 # return the list

    def CreateFileobjs (self):
        """ Create list of File Objects """
        fileobjects = []                        # list of all file objects
        for file in self.files:                 # each CSV file
            fullpath = os.path.join(self.readPath,file) # make full path str
            frame = pd.read_csv(fullpath,index_col=False)   # load in CSV
            frame = frame.to_numpy()                    # make np arr   
            for row in frame:                           # each row
                # 'FileObject' class is defined above
                fileobjects.append(FileObject(row))    # add row to obj list
            del(frame)                          # del frame  
        return fileobjects                      # return list of insts

class ProgramFinisher :
    """
    Handel all final Housekeeping bits of this program
        Ensure Everything has Run properly
    """
    def __init__(self,):
        """ Initialize ProgramFinisher Instance """
        dt_obj = datetime.datetime.now()
        starttime = dt_obj.isoformat(sep='.',timespec='auto')
        self.endtime = starttime.replace(':','.').replace('-','.')
        
    def __Call__(self,startime):
        """ Run Program Finisher Instance """
        print("Program Finish:",self.endtime)
        # More things will go here soon