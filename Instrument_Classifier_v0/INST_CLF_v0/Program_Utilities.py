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

            #### CLASS OBJECT DEFINITIONS ####

class file_object ():
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
        self.target_int = datarow[1]    # target as int  
        self.target_str = datarow[2]    # target as str
        dir_tree = self.fullpath.split('/')
        self.filename = dir_tree[-1]    # filename
        self.features = np.array([])    # feature vector
        # assign target for design matrix
        self.target = self.assign_target(self.target_int)
        
    def assign_target (self,target):
        """ Assign Target value to instance """
        self.target = target    # set y
        return self             # return self

    def read_audio (self):
        """ Read raw data from local path """
        rate,data = sciowav.read(self.fullpath)
        self.rate = rate            # set sample rate
        data = data.reshape(1,-1)   # flatten waveform
        self.waveform = data/np.abs(np.max(data))
        self.n_samples = len(self.waveform)
        return self             # return self

    def add_features (self,x):
        """ Add object x to feature vector attribute"""
        self.features = np.append(self.features,x)
        return self             # return self

    def __getfeatures__ (self):
        """ Assemble all features into single vector """
        return self.features    # return feature vector

    def __delfeatures__ (self):
        """ Delete all features (Save RAM) """
        pass

            #### FUNCTION DEFINITIONS ####

def Argument_Parser():
    """
    Create Argumnet Parser object or CLI
    --------------------------------
    *no args
    --------------------------------
    Return argument parser object
    """
    parser = argparse.ArgumentParser(prog='Instrument Classifier v0',
                                     usage='Classify .WAV files by instrument.',
                                     description="\n\t CLI Help for Instrument Classifier Program:",
                                     add_help=True)
    #parser.add_argument('-data_path',type=str,
    #                    help="Full Local Data Directory Path. Should have form \n\t \
    #                            C:/Users/yourname/.../my_data")
    parser.add_argument('-trgt_path',type=str,
                        help="Full Local Directory Path of file(s) containing \
                                rows of of data formatted: \
                                | Index | Fullpath  | Target Int    | Target Str |. \
                                Should have form \n\t  \
                                C:/Users/yourname/.../answers")
    parser.add_argument('-extr_path',type=str,
                        help="Full Local Data Directory Path to store intermediate \
                                file data. Reccommend using empty/new path.  Should have form \n\t \
                                C:/Users/yourname/.../answers/42.csv")
    # Parse and return args
    args = parser.parse_args()
    return args.trgt_path,args.extr_path

def create_fileobjs (filepath):
    """
    Load in locally stored target CSV as dataframe
    --------------------------------
    filepath (str) : Fill local directory path + filename
        Expected format:
            | Index | Fullpath  | Target Int    | Target Str |
    --------------------------------
    Return list of initialize class instances
    """
    frame = pd.read_csv(filepath,index_col=0)   # load in CSV
    frame = frame.to_numpy()                    # make np arr
    fileobjects = []                            # ist of all files
    for row in frame:
        fileobjects.append(file_object(row))    
    del(frame)                          # del frame
    return fileobjects                  # return list of insts

def Read_Directory_Tree (path,ext):
    """
    Read through directory and create instance of every file with matching extension
    --------------------------------
    path (str) : file path to read data from
    ext (str) : extension for appropriate file types
    --------------------------------
    Return list of wavfile class instances
    """
    file_objs = []                              # list to hold valid file
    for roots,dirs,files in os.walk(path):      # walk through the tree
        for file in files:                      # for each file
            if file.endswith(ext):              # matching extension
                file_objs.append(wavfile(file)) # add instance to list 
    return file_objs                            # return list of instances

def Validate_Directories (trgt_path,extr_path):
    """
    Check in passed directories are valid for program execution
    --------------------------------
    trgt_path (str) : Path and name of file containing rows of {name,target} pairs.
    extr_path (str) :  Path to store intermediate file data
    --------------------------------
    Return True, Terminate if fail ir
    """
    if os.path.isdir(trgt_path) == False:    # not not dir:
        print("\n\tERROR! - Cannot Locate:\n\t\t",trgt_path)
    os.makedirs(extr_path,exist_ok=True)    # create path for intermediate data
    return True                             # exist

