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

import Plotting_Utilities as plot_utils

"""
Program_Utilities.py - "Program Utilities"
    Contains Variables, Classes, and Definitions for Lower program functions
    Backends, Data structure objects, os & directory organization and validations

"""

            #### VARIABLE DECLARATIONS ####


            #### CLASS OBJECT DEFINITIONS ####

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
        self.target_int = datarow[1]    # target as int  
        self.target_str = datarow[2]    # target as str
        dir_tree = self.fullpath.split('/')
        self.filename = dir_tree[-1]    # filename
        # assign target for design matrix   
        self.target = self.target_int   # set target value

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

class Design_Matrix ():
    """
    Construct design-matrix-like object
    --------------------------------
    target (int) : Integer target value
    ndim (int) : Number of dimensions in this array
        2 - 2D matrix, used for MLP classifier 
            (n_samples x n_features)
        3 - 3D matrix, used for Spectrogram 
            (n_samples x n_rows x n_cols)
        4 - 4D matrix, used for phase-space
            (n_samples x n_rows x r_cols x n_
    --------------------------------
    Return instantiated feature_array instance
    """

    def __init__(self,ndim=2):
        """ Initialize Object Instance """
        self.X = []         # empty data structure
        self.shapes = []    # store explicit shapes of each samples
        self.ndim = ndim    # number of dimensions in array
        self.n_samples = 0  # no samples in design matrix

    def set_targets (self,y):
        """ Create 1D target array corresponding to sample class """
        self.targets = np.array(y)  
        return self

    def add_sample (self,x):
        """ Add features 'x' to design matrix, preserve shape """
        self.X.append(x.__getfeatures__())      # add sample to design matrix
        self.shapes.append(x.__getshape__())    # store shape       
        self.n_samples += 1                     # current number of samples
        return self
 
    def pad_2D (self,new_shape,offsets=(0,0)):
        """ Zero-Pad 2D samples to meet shape """
        new_X = np.zeros(shape=(self.n_samples,new_shape[0],new_shape[1]))   # create new design matrix
        for i in range(self.n_samples):     # iterate by sample
            dx,dy = offsets[0],offsets[1]   # align upper left     
            try: 
                new_X[i][dx:dx+self.X[i].shape[0],dy:dy+self.X[i].shape[1]] += self.X[i] 
            except:
                slice = self.X[i][:new_shape[0],:new_shape[1]]
                shape_diff = np.array(new_shape) - slice.shape      # needed padding
                slice = np.pad(slice,[[0,shape_diff[0]],[0,shape_diff[1]]])
                new_X[i] += slice
            self.shapes[i] = new_shape      # reset shape
        self.X = new_X              # overwrite
        self.X = self.X.reshape(self.n_samples,new_shape[0],new_shape[1],1)
        return self                         # return new instance

    def shape_by_sample (self,shape=None):
        """ Reshape design matrix by number of samples """
        if shape:
            self.X = self.X.reshape(shape)
        else:
            self.X = np.array(self.X).reshape(self.n_samples,-1)
        return self

    def scale_X (self,scaler):
        """ Apply standard preprocessing scaling to self.X """
        assert type(self.X) == np.ndarray
        return self

    def get_dims (self):
        """ get number of dimesnesion in this design matrix """
        return self.ndim
           
    def __getmatrix__(self):
        """ return design matrix as rect. np array """
        return self.X


class Feature_Array ():
    """
    Create Feature vector object
    --------------------------------
    target (int) : Integer target value
    --------------------------------
    Return instantiated feature_array instance
    """

    def __init__(self,target):
        """ Initialize Object Instance """
        self.target = target            # set target
        self.features = np.array([])    # arr to hold features

    def add_features (self,x,axis=None):
        """ Add object x to feature vector attribute"""
        self.features = np.append(self.features,x,axis=axis)
        return self             # return self

    def set_features (self,x):
        """ Clear feature array, reset to object 'x' - preserve shape """
        self.features = x
        return self

    def reshape_arr (self,new_shape=(1,-1)):
        """ Reshape feature array to 'new_shape' """
        self.features = self.features.reshape(new_shape)
        return self

    def set_attributes (self,names=[],attrbs=[]):
        """ Set additional attributes """
        for i,j in zip(names,attrbs):
            setattr(self,str(i),j)  
        return self

    def __getshape__(self):
        """ Return shape of feature attrb as tuple """
        return self.features.shape

    def __getfeatures__ (self):
        """ Assemble all features into single vector """
        return self.features    # return feature vector

    def __delfeatures__ (self):
        """ Delete all features (Save RAM) """
        del(self.features)      # delete all features from array
        return self             # return new self

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
                                     usage='Classify .WAV files by using pre-exisiting classifiered samples.',
                                     description="\n\t CLI Help for Instrument Classifier Program:",
                                     add_help=True)
    #parser.add_argument('-data_path',type=str,
    #                    help="Full Local Data Directory Path. Should have form \n\t \
    #                            C:/Users/yourname/.../my_data")
    parser.add_argument('data_path',type=str,
                        help="Full Local Directory Path of file(s) containing \
                                rows of of answer-key-like data; formatted: \
                                | Index | Fullpath  | Target Int    | Target Str |")
    parser.add_argument('model_path',type=str,
                        help="Full Local Data Directory Path to store intermediate \
                                file data. Reccommend using empty/new path.")
    # Parse and return args
    args = parser.parse_args()
    return args.data_path,args.model_path

def Create_Fileobjs (filepath):
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
    for row in frame:                           # each row
        fileobjects.append(File_Object(row))    
    del(frame)                          # del frame
    return fileobjects                  # return list of insts

def Directory_Map (keys,vals):
    """
    Create map (dictionary) of all local paths needed.
    --------------------------------
    keys (iter) : Iterable containing keys for dictionary (1 x M)
    vals (iter) : Iterable containing valus for dictionary (1 x M)
    --------------------------------
    """
    assert len(keys) == len(vals)   # must have same num pts
    map = {}                        # empty dictionary
    for key,val in zip(keys,vals):  # each key-val pair
        map.update({key:val})       # update dict
    return map                      # return the map

def Generate_Network (model,name,params,path):
    """
    Wrapper to Generate Tensorflow Sequenatial Model
        Handels args, local save path, and general housekeeping
    --------------------------------
    model (func) : Callable model from this script indicating model type to use
    name (str) : name to attatch to the model
    params (misc) : Misc. params to pass to 'model' function type
    path (str) : Local parent path to store model data
    --------------------------------

    """
    name = name.upper()                     # make name uppercase if not
    return None

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