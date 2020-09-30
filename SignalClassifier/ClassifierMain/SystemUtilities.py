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

        # Set Target for this File
        try:
            self.target = int(datarow[1])   # target as int           
        except:
            self.target = None              # no label
        
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

    def __init__(self):
        """ Intitialize CategoryDictionary Instance """
        raise NotImplementedError()
        
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

    def __init__(self,pathList=None,mode=None,newModels=None):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        self.starttime = dt_obj.isoformat(sep='_',timespec='auto').replace(':','.')
        print("Time Stamp:",self.starttime)
        self.programMode = mode
        self.newModels = newModels
        try:
            inputArgs = self.Argument_Parser()  # Parse Input args
            self.readPath  = inputArgs[0]       # Data files kept here 
            self.modelPath = inputArgs[1]       # store Network Model data
            self.exportPath = inputArgs[2]      # store network output
            self.programMode = inputArgs[3]     # set program mode
            self.newModels = inputArgs[4]       # create new models?
        except:
            self.readPath = pathList[0]
            self.modelPath =  pathList[1]
            self.exportPath = pathList[2]  
        assert self.programMode in ['train','train-predict','predict']
        assert self.newModels in [True,False]

        if (self.programMode == 'predict') and (self.newModels == True):
            print("\n\tERROR! -  Cannot run predictions on Untrained Models!")
            raise BaseException()

    def __repr__(self):
       """ Return String representation of Object/Instance """
       return "ProgramInitializer performs preprocessing for program parameters "

    def __Call__(self):
        """ Run Program Start Up Processes """        
        self.files = self.CollectCSVFiles()        # find CSV files
        fileobjects = self.CreateFileobjs()    # file all files
        self.n_files = len(fileobjects)        
        if self.programMode in ['train','train-predict']:
            self.n_classes = self.GetNClasses(fileobjects)
        else:
            self.n_classes = None
        self.StartupMesseges           # Messages to User
        fileobjects = np.random.permutation(fileobjects)    # permute
        return fileobjects,self.n_classes
            
    @property
    def StartupMesseges (self):
        """ Print out Start up messeges to Console """
        print("Running Main Program.....")
        print("\tCollecting data from:",self.readPath)
        print("\tStoring/Loading models from:",self.modelPath)
        print("\tExporting Predictions to:",self.exportPath)
        print("\tCreating new models?",self.newModels)
        print("\t\tFound",self.n_files,"files to read")
        print("\t\tFound",self.n_classes,"classes to sort")
        print("\n")

    def ArgumentParser(self):
        """ Process Command Line Arguments """
        parser = argparse.ArgumentParser(prog='SignalClassifier',
                                         usage='Classify .WAV files by using pre-exisiting classifiered samples.',
                                         description="\n\t CLI Help for Instrument Classifier Program:",
                                         add_help=True)

        parser.add_argument('data_path',type=str,
                            help="Full Local Directory Path of file(s) containing \
                                    rows of of answer-key-like data; formatted: \
                                    | Index | Fullpath  | Target Int    | Target Str |")
        parser.add_argument('model_path',type=str,
                            help="Full Local Data Directory Path to store intermediate \
                                    file data. Reccommend using empty/new path.")
        parser.add_argument('export_path',type=str,
                            help='Full Local Directory Path to export model predicitions and \
                                Evaluations to.')
        parser.add_argument('program_mode',type=str,
                            help="Mode for program execution. Must be in \
                                    ['train','train-test','predict']")
        parser.add_argument('new_models',type=bool,
                            help="If True, Networks sharing the same name are overwritten, \
                                and new models are created in place")
        # Parse and return args
        args = parser.parse_args()
        return [args.data_path,args.model_path,args.export_path,
                args.programMode,args.newModels]

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
        y = [x.target for x in fileobjects]   # collect target from each file
        try:                        # Attempt
            n_classes = np.amax(y)  # maximum value is number of classes
            return n_classes + 1    # account for zero-index
        except Exception:           # failure?
            return None             # no classes?


