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


import scipy.io.wavfile as sciowav


"""
Program_Utilities.py - "Program Utilities"
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
        self.fullpath = datarow[0]      # set full file path
        try:
            self.target = int(datarow[1])   # target as int           
        except:
            self.target = None              # no label
        dir_tree = self.fullpath.split('/')
        self.filename = dir_tree[-1]    # filename

    def AssignTarget (self,target):
        """ Assign Target value to instance """
        self.target = target    # set y
        return self             # return self

    def ReadAudio (self):
        """ Read raw data from local path """
        rate,data = sciowav.read(self.fullpath)    
        data = data.reshape(1,-1).ravel()   # flatten waveform
        self.rate = rate            # set sample rate
        self.waveform = data/np.abs(np.amax(data))
        self.n_samples = len(self.waveform)
        return self             # return self

            #### PROGRAM PROCESSING CLASSES ####

class ProgramStart:
    """
    Object to handle all program preprocessing
    --------------------------------
    readpath (str) : Local path nagivating to where data is stored
    modelpath (str) : Local path to store data related to Nerual network Models
    exportpath (str) : Local path to export final information
    mode (str) : String indicating which mode to execute program with
    newmodels (bool): If True, create new Nueral Network Models
    --------------------------------
    Return Instantiated Program Start Class Instance
    """

    def __init__(self,readpath=None):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        self.starttime = dt_obj.isoformat(sep='_',timespec='auto').replace(':','.')
        self.readpath = readpath
        print("Time Stamp:",self.starttime)
        
            
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("Running Main Program.....")
        print("\t\tFound",self.n_files,"files to read")
        print("\t\tFound",self.n_classes,"classes to sort")
        print("\n")

    def __call__(self):
        """ Run Program Start Up Processes """        
        self.files = self.CollectCSVFiles()        # find CSV files
        fileobjects = self.CreateFileobjs()    # file all files
        self.n_files = len(fileobjects)        
        self.n_classes = self.GetNClasses(fileobjects)
        self.StartupMesseges           # Messages to User
        fileobjects = np.random.permutation(fileobjects)    # permute
        return fileobjects,self.n_classes

    def CollectCSVFiles (self,exts='.csv'):
        """ Walk through Local Path and File all files w/ extension """
        csv_files = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):       
                    csv_files.append(file)
        return csv_files

    def CreateFileobjs (self):
        """ Create list of File Objects """
        fileobjects = []                        # list of all file objects
        for file in self.files:                 # each CSV file
            fullpath = os.path.join(self.readpath,file) # make full path str
            frame = pd.read_csv(fullpath,index_col=0)   # load in CSV
            frame = frame.to_numpy()                    # make np arr   
            for row in frame:                           # each row
                # 'FileObject' class is defined above
                fileobjects.append(FileObject(row))    # add row to obj list
            del(frame)                          # del frame  
        return fileobjects                      # return list of insts

    def GetNClasses (self,fileobjects):
        """ Find Number of classes in target vector """
        y = [x.target for x in fileobjects]   # collect target from each file
        try:                        # Attempt
            n_classes = np.amax(y)  # maximum value is number of classes
            return n_classes + 1    # account for zero-index
        except Exception:           # failure?
            return None             # no classes?



