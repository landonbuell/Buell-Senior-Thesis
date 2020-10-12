"""
Landon Buell
PHYS 799
Classifier Cross-Validation
5 October 2020
"""

            #### IMPORTS ####

import numpy as np
import pandas as pd
import os
import sys
import datetime
import argparse

import scipy.io.wavfile as sciowav


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

    def __Call__(self,fileObjs):
        """ Run Category Encode / Decode Dictionary """   
        # A new Model is created, Overwrite existing File
        print("\tBuilding new Encode/Decode Dictionary...")
        self.encoder,self.decoder = self.BuildCategories(fileObjs)
        self.ExportCategories()        
        return self

    def LoadCategories (self):
        """ Load File to Match Int -> Str Class """
        decoder = {}
        encoder = {}
        rawData = pd.read_csv(self.filePath)
        Ints,Strs = rawData.iloc[:,0],rawData.iloc[:,1]
        for Int,Str in zip(Ints,Strs):      # iterate by each
            encoder.update({str(Str):int(Int)})
            decoder.update({int(Int):str(Str)})
        return encoder,decoder

    def ExportCategories (self):
        """ Export Decode Dictionary (Int -> Str) """
        decodeList = sorted(self.decoder.items())
        cols = ["Target Int","Target Str"]
        decodeFrame = pd.DataFrame(data=decodeList,columns=cols)
        decodeFrame.to_csv(self.filePath,index=False)
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
        # Organize in numerical order
        sortedItems = sorted(decoder.items())
        encoder = {}      # reset encoder
        for (targetInt,targetStr) in sortedItems:
            encoder.update({targetStr:targetInt})
            decoder.update({targetInt:targetStr})
        return encoder,decoder
             
            #### PROGRAM PROCESSING CLASS ####

class ProgramInitializer:
    """
    Object to handle all program preprocessing
    --------------------------------
    pathList (list) : List of Local Directory paths - 
        readPath - local path where target label files are stored
        modelPath - local path where model data can be exported to 
        exptPath - local path where X-Validation data be exported to
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,pathList=[None,None,None],modelName=None):
        """ Initialize Class Object Instance """
        self.readPath = pathList[0]  
        self.modelPath = pathList[1]  
        self.exportPath = pathList[2]
        self.modelName = modelName

    def __repr__(self):
       """ Return String representation of Object/Instance """
       return "ProgramInitializer performs preprocessing for program parameters "

    def __Call__(self):
        """ Run Program Start Up Processes """     
        print("\nRunning Main Program.....\n")
        self.files = self.CollectCSVFiles()     # find CSV files
        fileObjects = self.CreateFileobjs()     # file all files
        self.n_files = len(fileObjects)         # get number of files
        
         # Construct Encoder / Decoder Dictionaries
        self.categories = CategoryDictionary(self.modelPath,self.modelName)
        self.categories.__Call__(fileObjects)
        self.n_classes = self.GetNClasses(fileObjects)
     
        # Final Bits
        self.StartupMesseges
        fileObjects = np.random.permutation(fileObjects)    # permute
        return fileObjects

    @property
    def GetLocalPaths (self):
        """ Return Necessaru Directory Paths """
        return (self.readPath,self.exportPath)

    @property
    def GetDecoder (self):
        """ Return Decoder Dictionary, maps Int -> Str """
        return self.categories.decoder
            
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("\tCollecting data from:\n\t\t",self.readPath)
        print("\tStoringModels to:\n\t\t",self.modelPath)
        print("\tExporting Predictions to:\n\t\t",self.exportPath)
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
        try:                        # Attempt    
            return np.max(y) + 1    # account for zero-index
        except Exception:           # failure?
            return None             # no classes?


