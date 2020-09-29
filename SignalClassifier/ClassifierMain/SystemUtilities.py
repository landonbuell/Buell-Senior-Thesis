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
        
    def AssignTarget (self,target):
        """ Assign Target value to instance """
        self.target = target    # set y
        return self             # return self

    def ReadFileData (self):
        """ Read Data into array based on data type """
        if self.extension == ".wav":    # is wav file?
            return self.ReadFileWAV()   # read 
        elif self.extension == ".txt":  # is txt file?
            return self.ReadFileTXT()   # read .txt file
        else:
            raise NotImplementedError() # extension not matched

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
            self.cols = ["Epoch Num","Loss Score","Precision","Recall"]
        elif programMode == "Predict":  # output for predictions
            self.cols = ["Filepath","Label","Prediction"]
        else:
            print("\n\tERROR! - File mode not recognized!")
            self.cols = []
            raise BaseException()
        self.exportPath = exportPath
        self.InitData()
    
    def InitData (self):
        """ Initialize Output Structure """
        self.Data = pd.DataFrame(data=None,columns=self.cols)   # create frame
        self.Data.to_csv(path_or_buf=self.exportPath)           # export frame
        return self

    def UpdateData (self,X):
        """ Update Data in Output Frame """
        frame = pd.DataFrame(data=X)        # create new dataframe
        frame.to_csv(path_or_buf=self.exportPath,columns=self.cols,
                     header=False,mode='a') # append the output frame
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

    def __Call__(self,newModel,fileObjs):
        """ Run Category Encode / Decode Dictionary """
        if newModel == True:        
            # A new Model is created, Overwrite existing File
            print("\t\tBuilding new Encode/Decode Dictionary...")
            self.encoder,self.decoder = self.BuildCategories(fileObjs)
            self.ExportCategories()
        else:
            # Check if the encoder / Decoder Exists:
            if os.path.isfile(self.filePath) == True:
                print("\t\tFound Encode/Decode Dictionary")
                # the file exists, load it as enc/dec dict
                self.encoder,self.decoder = self.LoadCategories()
            else:
                # File does not exists, make a new Dictionary
                print("\t\tCount not fine Encode/Decode Dictionary, building new")
                self.encoder,self.decoder = self.BuildCategories(fileObjs)
                self.ExportCategories()
        return self

    def LoadCategories (self):
        """ Load File to Match Int Class to Str Class """
        data = pd.read_csv(self.fileName,header=0,usecols=[1,2])
        raise NotImplementedError()
        return None,None

    def ExportCategories (self):
        """ Export Decode Dictionary (Int -> Str) """
        raise NotImplementedError()
        return self

    def BuildCategories (self,fileObjs):
        """ Construct Dictionary to map STR -> INT """
        encoder = {}            # empty enc dict
        decoder = {}            # empty dec dict
        targetInts = [x.targetInt for x in fileObjs]
        targetStrs = [x.targetStr for x in fileObjs]
        for x,y in zip(targetInts,targetStrs):
            if x not in encoder.keys():     
                decoder.update({x:y})   # add int : str pair
                encoder.update({y:x})   # add str : int pair
        return encoder,decoder
             
            #### PROGRAM PROCESSING CLASS ####

class ProgramInitializer:
    """
    Object to handle all program preprocessing
    --------------------------------
    pathList (list) : List of Local Directory paths - 
        [*readPath - training data index, *modelpath - Store models here,
            *exportPath - predictions / traing data are exported to
    mode (str) : String indicating which mode to execute program with
    newmodels (bool): If True, create new Nueral Network Models
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,pathList=[None,None,None],mode=None,
                 modelName=None,newModel=None):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        self.starttime = \
            dt_obj.isoformat(sep='.',timespec='auto').replace(':','.').replace('-','.')
        print("Time Stamp:",self.starttime)
        self.programMode = mode
        self.newModel = newModel
        self.modelName = modelName
        try:
            inputArgs = self.ArgumentParser()   # Parse Input args
            self.readPath  = inputArgs[0]       # Data files kept here 
            self.modelPath = inputArgs[1]       # store Network Model data
            self.exportPath = inputArgs[2]      # store network output
            self.programMode = inputArgs[3]     # set program mode
            self.modelName = inputArgs[4]       # name to use
            self.newModel = inputArgs[5]        # create new model?         
        except:
            self.readPath = pathList[0]
            self.modelPath =  pathList[1]
            self.exportPath = pathList[2]  
        # Ensure that all variables make sense
        assert self.programMode in ['train','train-predict','predict']
        assert self.newModel in [True,False]
        assert modelName is not None

        if (self.programMode == 'predict') and (self.newModel == True):
            print("\n\tERROR! -  Cannot run predictions on an untrained (new) Model!")
            raise BaseException()

    def __repr__(self):
       """ Return String representation of Object/Instance """
       return "ProgramInitializer performs preprocessing for program parameters "

    def __Call__(self):
        """ Run Program Start Up Processes """        
        self.files = self.CollectCSVFiles()     # find CSV files
        fileObjects = self.CreateFileobjs()     # file all files
        self.n_files = len(fileObjects)         # get number of files
        
        # Construct Encoder / Decoder Dictionaries
        self.categories = CategoryDictionary(self.modelPath,self.modelName)
        self.categories.__Call__(self.newModel,fileObjects)

        # Find Number of Classes
        if self.programMode in ['train','train-predict']:
            self.n_classes = self.GetNClasses(fileObjects)  
        else:
            self.n_classes = None 

        # Final Bits
        self.StartupMesseges           # Messages to User
        fileObjects = np.random.permutation(fileObjects)    # permute
        return fileObjects

    @property
    def GetLocalaths (self):
        """ Return Necessaru Directory Paths """
        return (self.readPath,self.exportPath,self.modelPath)

    @property
    def GetModelParams (self):
        """ Return Important Parameters for Creating Models """
        return (self.modelName,self.newModel,self.n_classes,self.starttime)
            
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("\nRunning Main Program.....")
        print("\tCollecting data from:\n\t\t",self.readPath)
        print("\tStoring/Loading models from:\n\t\t",self.modelPath)
        print("\tExporting Predictions to:\n\t\t",self.exportPath)
        print("\tModel name:",self.modelName)
        print("\tCreating new models?",self.newModel)
        print("\tNumber of Files found:",self.n_files)
        print("\tNumber of Categories found:",self.n_classes)
        print("\n")
        return None

    def ArgumentParser(self):
        """ Process Command Line Arguments """
        parser = argparse.ArgumentParser(prog='SignalClassifier',
                                         usage='Classify .WAV files by using pre-exisiting classifiered samples.',
                                         description="\n\t CLI Help for Instrument Classifier Program:",
                                         add_help=True)

        parser.add_argument("dataPath",type=str,
                            help="Full Local Directory Path of file(s) containing \
                                    rows of of answer-key-like data; formatted: \
                                    | Index | Fullpath  | Target Int    | Target Str |")
        parser.add_argument('modelPath',type=str,
                            help="Full Local Data Directory Path to store intermediate \
                                    file data. Reccommend using empty/new path.")
        parser.add_argument('exportPath',type=str,
                            help='Full Local Directory Path to export model predicitions and \
                                Evaluations to.')
        parser.add_argument('programMode',type=str,
                            help="Mode for program execution. Must be in \
                                    ['train','train-test','predict']")
        parser.add_argument("modelName",type=str,
                            help="Name of Model to use or create")
        parser.add_argument("newModels",type=bool,
                            help="If True, Networks sharing the same name are overwritten, \
                                and new models are created in place")
        # Parse and return args
        args = parser.parse_args()
        return [args.dataPath,args.modelPath,args.exportPath,
                args.programMode,args.modelName,args.newModels]

    def CollectCSVFiles (self,exts='.csv'):
        """ Walk through Local Path and File all files w/ extension """
        csv_files = []
        for roots,dirs,files in os.walk(self.readPath):  
            for file in files:                  
                if file.endswith(exts):       
                    csv_files.append(file)
        return csv_files

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

    def GetNClasses (self,fileobjects):
        """ Find Number of classes in target vector """
        y = [x.targetInt for x in fileobjects]   # collect target from each file
        try:                            # Attempt    
            return len(np.unique(y))    # account for zero-index
        except Exception:               # failure?
            return None                 # no classes?


