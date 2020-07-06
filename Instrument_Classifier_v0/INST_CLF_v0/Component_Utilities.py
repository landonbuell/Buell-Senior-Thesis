"""
Landon Buell
PHYS 799
Instrument Classifier v0
12 June 2020
"""

            #### IMPORTS ####

import numpy as np
import sys
import os

import Program_Utilities as prog_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models 

"""
Component_Utilities.py - 'Component Utilities'
    Contains Definitions that are only called directly from MAIN script
    Functions are large & perform Groups of important operations
"""

            #### FUNCTION DEFINITIONS ####  

def organize_paths (model_names):
    """
    Initialize all local directory path variables and create Neural Network Models
    --------------------------------
    model_names (iter) : Iterable containing names for 3 classifier models
    --------------------------------
    Return map (dict) of local paths
    """
    assert len(model_names) == 3                            # one name per model
    init_path = os.getcwd()         # current path
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classfier_v0'
    data_path = os.path.join(parent,'Target-Data')  # target vectors are stored here
    model_path = os.path.join(parent,'Model-Data')   # Saved models are stored here
    #data_path,model_path = prog_utils.Argument_Parser()

    prog_utils.Validate_Directories(must_exist=[data_path],
                                    must_create=[model_path])

    # Initialize Paths Dictionary
    directory_map = {'HOME':init_path,'DATA':data_path,'MODELS':model_path}
    for name in model_names:            # each model name
        key,val = name.upper(),os.path.join(model_path,name.upper())
        directory_map.update({key:val})

    return directory_map 

def 

