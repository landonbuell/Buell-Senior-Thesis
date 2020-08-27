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

import Plotting_Utilities as plot_utils

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

class OutputStructure :
    """
    Data Structure to Store and Output History or Prediction arrays
    """

    def __init__(self,programMode,exportPath):
        """ Initialize Class Object instance """
        if programMode == "Train":      # output for training history
            self.keys = ["Epoch Num","Loss Score","Precision","Recall"]
        elif programMode == "Predict":  # output for predictions
            self.keys = ["Filepath","Label","Prediction"]
        else:
            print("\n\tERROR! - File mode not recognized!")
            self.keys = []
            raise BaseException()
        self.exportPath = exportPath
        self.InitData()
    
    def InitData (self):
        """ Initialize Output Structure """
        self.Data = pd.DataFrame(data=None,columns=self.keys)   # create frame
        self.Data.to_csv(path_or_buf=self.exportPath)           # export frame
        return self

    def UpdateData (self,X):
        """ Update Data in Output Frame """
        frame = pd.DataFrame(data=X)        # create new dataframe
        frame.to_csv(path_or_buf=self.exportPath,columns=self.keys,
                     header=False,mode='a') # append the output frame
        return self
        

            #### PROGRAM PROCESSING CLASS ####

class ProgramInitializer:
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

    def __init__(self,readpath=None,modelpath=None,exportpath=None,
                 mode=None,newmodels=None):
        """ Initialize Class Object Instance """
        dt_obj = datetime.datetime.now()
        self.starttime = dt_obj.isoformat(sep='_',timespec='auto').replace(':','.')
        print("Time Stamp:",self.starttime)
        # If arguments given:
        if readpath is not None:
            self.readpath = readpath
        if modelpath is not None: 
            self.modelpath = modelpath
        if exportpath is not None:
            self.exportpath = exportpath
        if mode is not None:
            self.program_mode = mode
        if newmodels is not None:
            self.new_models = newmodels
        try:
            input_args = self.Argument_Parser() # Parse Input args
            self.readpath  = input_args[0]      # Data files kept here 
            self.modelpath = input_args[1]      # store Network Model data
            self.exportpath = input_args[2]     # store network output
            self.program_mode = input_args[3]   # set program mode
            self.new_models = input_args[4]     # create new models?
        except:
            pass
        assert self.program_mode in ['train','train-predict','predict']

        if (self.program_mode == 'predict') and (self.new_models == True):
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
        if self.program_mode in ['train','train-test']:
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
        print("\tCollecting data from:",self.readpath)
        print("\tStoring/Loading models from:",self.modelpath)
        print("\tExporting Predictions to:",self.exportpath)
        print("\tCreating new models?",self.new_models)
        print("\t\tFound",self.n_files,"files to read")
        print("\t\tFound",self.n_classes,"classes to sort")
        print("\n")

    def ArgumentParser(self):
        """ Process Command Line Arguments """
        parser = argparse.ArgumentParser(prog='Instrument Classifier v0',
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
                args.program_mode,args.new_models]

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

    def ValidateDirectories (must_exist=[],must_create=[]):
        """
        Check in passed directories are valid for program execution
        --------------------------------
        must (str) : Path and name of file containing rows of {name,target} pairs.
        extr_path (str) :  Path to store intermediate file data
        --------------------------------
        Return True, Terminate if fail ir
        """
        for path in must_exist:                 # paths that must exisit
            if os.path.isdir(path) == False:    # not not dir:
                print("\n\tERROR! - Cannot Locate:\n\t\t",path)
        for path in must_create:                # path that must create
            os.makedirs(path,exist_ok=True)     # create path
        return None

