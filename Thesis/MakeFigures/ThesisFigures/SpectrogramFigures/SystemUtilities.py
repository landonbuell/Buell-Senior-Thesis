"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import pandas as pd

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

    def __init__(self,readPath):
        """ Initialize Class Object Instance """

        # Establish Paths
        self.readPath = readPath
       
    def __repr__(self):
       """ Return String representation of Object/Instance """
       return "ProgramInitializer performs preprocessing for program parameters "

    def __Call__(self):
        """ Run Program Start Up Processes """     
        print("\nRunning Main Program.....\n")
        self.files = self.CollectFiles()        # find CSV files
        fileObjects = self.CreateFileobjs()     # file all files
        self.n_files = len(fileObjects)         # get number of files        
        fileObjects = np.random.permutation(fileObjects)    # permute
        return fileObjects

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
