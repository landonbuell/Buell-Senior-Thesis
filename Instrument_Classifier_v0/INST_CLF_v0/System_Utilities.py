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
import argparse

import scipy.io.wavfile as sciowav
from sklearn.model_selection import train_test_split

import Plotting_Utilities as plot_utils

"""
Program_Utilities.py - "Program Utilities"
    Contains Variables, Classes, and Definitions for Lower program functions
    Backends, Data structure objects, os & directory organization and validations

"""

            #### VARIABLE DECLARATIONS ####


            #### DATA STRUCTURE CLASSES ####

class File_Object ():
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

    def assign_target (self,target):
        """ Assign Target value to instance """
        self.target = target    # set y
        return self             # return self

    def read_audio (self):
        """ Read raw data from local path """
        rate,data = sciowav.read(self.fullpath)
        self.rate = rate            # set sample rate
        data = data.reshape(1,-1).ravel()   # flatten waveform
        self.waveform = data/np.abs(np.amax(data))
        self.n_samples = len(self.waveform)
        return self             # return self

            #### PROGRAM PROCESSING CLASSES ####

class Program_Start:
    """
    Handel all pre-processing for program
        - Command line Arguments
        - Setup Directory paths

    """

    def __init__(self,readpath=None,modelpath=None,mode=None,newmodels=None):
        """ Inititalize Program Attributes """
        # If arguments given:
        if readpath:
            self.readpath = readpath
        if modelpath: 
            self.modelpath = modelpath
        if mode:
            self.program_mode = mode
        if newmodels:
            self.new_models = newmodels
        else:                   # If ANY fail....
            input_args = self.Argument_Parser()     # Parse Input args
            self.readpath  = input_args[0]          # Data files kept here 
            self.modelpath = input_args[1]          # store Network Model data
            self.program_mode = input_args[2]       # set program mode
            self.new_models = input_args[3]         # create new models?
        assert self.program_mode in ['train','train-test','predict']

        if self.program_mode == 'predict' and self.new_models == True:
            print("\n\tERROR! -  Cannot run predictions on New, Untrained Models!")
            sys.exit()

    def __startup__(self):
        """ Run Program Start Up Processes """
        self.files = self.Collect_CSVs()        # find CSV files
        fileobjects = self.Create_Fileobjs()    # file all files
        self.n_files = len(fileobjects)
        print("\tFound",self.n_files,"files")

        if self.program_mode in ['train','train-test']:
            n_classes = self.get_nclasses(fileobjects)
            print("\tFound",n_classes,"classes")
        else:
            n_classes = None
            
        return fileobjects,n_classes

    def Argument_Parser():
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
        parser.add_argument('program_mode',type=str,
                            help="Mode for program execution. Must be in \
                                    ['train','train-test','predict']")
        parser.add_argument('new_models',type=bool,
                            help="If True, Networks sharing the same name are overwritten, \
                                and new models are created in place")
        # Parse and return args
        args = parser.parse_args()
        return [args.data_path,args.model_path,args.program_mode,args.new_models]

    def Collect_CSVs (self,exts='.csv'):
        """ Walk through Local Path and File all files w/ extension """
        csv_files = []
        for roots,dirs,files in os.walk(self.readpath):  
            for file in files:                  
                if file.endswith(exts):       
                    csv_files.append(file)
        return csv_files

    def Create_Fileobjs (self):
        """ Create list of File Objects """
        fileobjects = []                        # list of all file objects
        for file in self.files:                 # each CSV file
            fullpath = os.path.join(self.readpath,file) # make full path str
            frame = pd.read_csv(fullpath,index_col=0)   # load in CSV
            frame = frame.to_numpy()                    # make np arr   
            for row in frame:                           # each row
                fileobjects.append(File_Object(row))    # add row to obj list
            del(frame)                          # del frame  
        fileobjects = np.random.permutation(fileobjects)# permute
        return fileobjects                      # return list of insts

    def get_nclasses (self,fileobjects):
        """ Find Number of classes in target vector """
        y = [x.target for x in fileobjects]   # collect target from each file
        try:                        # Attempt
            n_classes = np.amax(y)  # maximum value is number of classes
            return n_classes + 1    # account for zero-index
        except Exception:           # failure?
            return None             # no classes?

    def Validate_Directories (must_exist=[],must_create=[]):
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

def Update_Map (map={},keys=[],vals=[]):
    """
    Update any map (dictionary) with keys and values
        New dictionary is create dif one is not provided
    --------------------------------
    map (dict) : Empty or existing dictionary object to populate
    keys (iter) : Iterable containing keys for dictionary (1 x M)
    vals (iter) : Iterable containing valus for dictionary (1 x M)
    --------------------------------
    """
    assert len(keys) == len(vals)   # must have same num pts
    for key,val in zip(keys,vals):  # each key-val pair
        map.update({key:val})       # update dict
    return map                      # return the map
