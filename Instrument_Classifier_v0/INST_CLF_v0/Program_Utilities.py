"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import os
import sys
import argparse

            #### CLASS OBJECT DEFINITIONS ####


                

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
    parser.add_argument('-data_path',type=str,
                        help="Full Local Data Directory Path. Should have form \n\t \
                                C:/Users/yourname/.../my_data")
    parser.add_argument('-trgt_path',type=str,
                        help="Full Local Directory Path of file(s) containing \
                                rows of {name,target} pairs. Should have form \n\t \
                                C:/Users/yourname/.../answers")
    parser.add_argument('-extr_path',type=str,
                        help="Full Local Data Directory Path to store intermediate \
                                file data. Reccommend using empty/new path.  Should have form \n\t \
                                C:/Users/yourname/.../answers/42.csv")
    # Parse and return args
    args = parser.parse_args()
    return args.data_path,args.tgrt_path,args.extr_path

def Validate_Directories (data_path,trgt_path,extr_path):
    """
    Check in passed directories are valid for program execution
    --------------------------------
    data_path (str) : Path where data files are stored
    trgt_path (str) : Path and name of file containing rows of {name,target} pairs.
    extr_path (str) :  Path to store intermediate file data
    --------------------------------
    Return True, Terminate if fail ir
    """
    for path in [data_path,trgt_path]:      # check paths
        if os.path.isdir(path) == False:    # not not dir:
            print("\n\tERROR! - Cannot Locate:\n\t\t",path)
            sys.exit()                      # terminate
    os.makedirs(extr_path,exist_ok=True)    # create path for intermediate data
    return True                             # exist

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