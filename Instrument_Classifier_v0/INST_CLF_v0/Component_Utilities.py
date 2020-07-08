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
    Initialize all local directory path variable
    --------------------------------
    model_names (iter) : Iterable containing names for 3 classifier models
    --------------------------------
    Return map (dict) of local paths
    """
    assert len(model_names) == 3                            # one name per model
    init_path = os.getcwd()         # current path
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
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

def create_models (model_names,map,n_classes,overwrite=[True,True,True]):
    """
    Initialize three Neural Network models
    --------------------------------
    model_names (iter) : Iterable containing names for 3 classifier models
    map (dict) : Directory Dictionary map, containin important local paths
    n_classes (int) : Number of unique target classes for classification
    overwrite (iter/bool) : Iterable of booleans or booleans indicating to overwrite
    --------------------------------
    Return None
    """
    assert len(model_names) == 3            # one name per model
    model_names = [name.upper() for name in model_names]

    # Create Multilayer Perceptron Model
    model = Neural_Network_Models.Multilayer_Perceptron(name=model_names[0],
                n_features=Neural_Network_Models.n_features,n_classes=n_classes,
               layerunits=[64,64])          # Create keras Model instance
    model.save(filepath=map[model_names[0]],overwrite=overwrite[0])   # Save instance locally
    del(model)                              # delete from RAM
    
    # Create Spectrogram Classifier Conv Network Model
    model = Neural_Network_Models.Convolutional_Neural_Network_2D(name=model_names[1],
            in_shape=Neural_Network_Models.sepectrogram_shape,n_classes=n_classes,filtersizes=[16,8],
           kernelsizes=[(3,3),(3,3)],poolsizes=[(2,2),(2,2)],layerunits=[64,64]) 
    model.save(filepath=map[model_names[1]],overwrite=overwrite[1])   # Save instance locally
    del(model)                              # delete from RAM


    # Create Phase-Space Classifier Conv Network Model
    model = Neural_Network_Models.Convolutional_Neural_Network_2D(name=model_names[2],
            in_shape=Neural_Network_Models.phasespace_shape,n_classes=n_classes,filtersizes=[8,8],
            kernelsizes=[3,3],poolsizes=[4,4],layerunits=[64])
    model.save(filepath=map[model_names[2]],overwrite=overwrite[2])   # Save instance locally
    del(model)                              # delete from RAM

    return None
    
def Act_on_Batch (FILE_BATCH,model_names,map,n_classes,mode='train'):
    """
    Extract features and train/test models from a batch of audio files
    --------------------------------
    FILE_BATCH (list) : List of file object instace to extract features from
    model_names (iter) : Iterable containing names for 3 classifier models
    map (dict) : Directory Dictionary map, containin important local paths
    n_classes (int) : Number of unique target classes for classification
    mode (str) : Indicates if batch is use to [train/test/predict]
    --------------------------------
    Return None
    """
    assert mode in ['train','test','predict']
    if mode != 'predict':           # not predicting (there is answer-key)
        Y = ML_utils.target_array(FILE_BATCH,n_classes,matrix=True)
        
    # Extract Design Matrices for each Model

    return None