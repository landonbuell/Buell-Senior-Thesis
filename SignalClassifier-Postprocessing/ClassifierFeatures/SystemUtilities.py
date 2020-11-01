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

import FeatureUtilities as feat_utils
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

class CategoryDictionary :
    """
    Category Dictionary Maps an integer class to string class
        and vice-versa
    """

    def __init__(self,localPath,modelName):
        """ Intitialize CategoryDictionary Instance """
        self.localPath = localPath
        self.fileName = modelName+"Categories.csv"
        self.filePath = os.path.join(self.localPath,self.fileName)

    def __Call__(self,fileObjs):
        """ Run Category Encode / Decode Dictionary """     
        # A new Model is created, Overwrite existing File
        print("\tBuilding new Encode/Decode Dictionary...")
        self.encoder,self.decoder = self.BuildCategories(fileObjs)
        self.ExportCategories()
        return self

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
    pathList (list) : List of 2 Local Directory paths:
        readPath - local path where target data is held       
        exportPath - local path where training history / predictions are exported
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,pathList=[None,None]):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        starttime = dt_obj.isoformat(sep='.',timespec='auto')
        self.starttime = starttime.replace(':','.').replace('-','.')
        print("Time Stamp:",self.starttime)

        # Establish Paths
        self.readPath = pathList[0]
        self.exportPath = pathList[1]
               
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
        self.categories = CategoryDictionary(self.exportPath,"Feature")
        self.categories.__Call__(fileObjects)

        decodeKeys = [i for i in self.GetDecoder.keys()]
        self.n_classes = np.max(decodeKeys) + 1

        filesByCategory = self.FilesByCategory(fileObjects)
          
        # Final Bits
        self.StartupMesseges           # Messages to User
        return filesByCategory

    def FilesByCategory (self,fileObjs):
        """ Group files intoa dictionary by category """
        #groupedFiles = {{x:[]} for x in range(0,self.n_classes)}
        groupedFiles = {}
        for i in range(self.n_classes):
            groupedFiles.update({i:[]})
        for file in fileObjs:
            groupedFiles[file.targetInt].append(file)
        return groupedFiles
       
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
        print("\tExporting Predictions to:\n\t\t",self.exportPath)
        print("\tNumber of Categories found:",self.n_classes)
        print("\n")
        return None

    @staticmethod
    def InitOutputMatrix(exportPath,n_features):
        """ """
        cols =  ["Target Str","Target Int"]
        cols += ["FTR"+str(i) for i in range(n_features)]
        outputFrame = pd.DataFrame(data=None,columns=cols)
        outputFrame.to_csv(exportPath,index=False,header=True)

    def CollectCSVFiles (self,exts='.csv'):
        """ Find .csv files in Local Path """
        csv_files = []
        for file in os.listdir(self.readPath):                       
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