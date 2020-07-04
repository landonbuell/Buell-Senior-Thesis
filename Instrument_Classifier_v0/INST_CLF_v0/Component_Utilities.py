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

def PATHS ():
    """
    Initialize all local directory path variables
    --------------------------------
    * no args
    --------------------------------
    Return map (dict) of local paths
    """
    init_path = os.getcwd()
    trgt_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0/TARGETS.csv'
    save_path = 'C:/Users/Landon/Documents/GitHub/Buell-Senior-Thesis/Instrument_Classifier_v0/Model-Data'
    #trgt_path,itmd_path = prog_utils.Argument_Parser()
    prog_utils.Validate_Directories(trgt_path,save_path) 
    MAP = prog_utils.Directory_Map(
        keys=['HOME','TARGET','MODELS'],
        vals=[init_path,trgt_path,save_path])
    return MAP

def MODELS_FILEOBJS (MAP,names,new_models=True,permute=True):
    """
    Construct 3 neural Network Models, and create local paths to save them
    --------------------------------
    MAP (dict) : dictionary containing important local path variables
    names (iter) : Iterable containing names for 3 classifier models
    new_models (bool) : If Truem new models are create to overwrite old ones
    permute (bool) : If True, order of file object instances are permuted
    --------------------------------
    Return map (dict) of local paths, list of Fileobject instances,
        One-hot-encoded target vector 
    """
    assert len(names) == 3              # 3 names for 3 models
    names = [x.upper() for x in names]  # make uppercase

    # Construct File-objects array
    FILEOBJS = prog_utils.Create_Fileobjs(MAP['TARGET'])
    if permute == True:                             # if permute...
        FILEOBJS = np.random.permutation(FILEOBJS)  # permute!
    TARGETS,n_classes = ML_utils.construct_targets(FILEOBJS)

    # Create Spectrogram Classifier Model
    model = Neural_Network_Models.Convolutional_Neural_Network_2D(names[1],
        in_shape=Neural_Network_Models.sepectrogram_shape,n_classes=n_classes,
        kernelsizes=[(3,3),(3,3)],poolsizes=[(2,2),(2,2)],layerunits=[128])
    save_path = os.path.join(MAP['MODELS'],'SPECTROGRAM')
    MAP.update({'SPECTROGRAM':save_path})
    model.save(save_path,overwrite=True)   

    # Create Multilayer Perceptron Model
    model = Neural_Network_Models.Multilayer_Perceptron(names[2],
            n_features=15,n_classes=n_classes,layerunits=[80,80])
    save_path = os.path.join(MAP['MODELS'],'PERCEPTRON')
    MAP.update({'PERCEPTRON':save_path})
    model.save(save_path,overwrite=True)
    
    return MAP,FILEOBJS,TARGETS

def TRAIN_on_SET (BATCH,Y,MAP):
    """
    Collect all features from a Subset of fileobjects and train models
    --------------------------------
    BATCH (iter) : Subset of file objects to train on (1 x n_samples)
    Y (arr) : One-hot target vector for samples (n_samples x n_classes)
    MAP (dict) : dictionary containing important local path variables
    --------------------------------
    Return None
    """
    print("\t\tProcessing Batch")
    DESIGN_MATRICIES = ML_utils.Design_Matrices(BATCH) # List of Design Matrix objs
    # 'X' is in order: [Phase-Space Clf, Spectrogram Clf, MLP Clf]
    MODELS = [MAP['SPECTROGRAM'],
              MAP['PERCEPTRON']]

    print("\t\tTraining Batch Set")
    for X,path in zip(DESIGN_MATRICIES,MODELS):
        # Load Model into RAM & Train
        model = Neural_Network_Models.keras.models.load_model(path)
        print("\t\t\tTraining Model:",model.name)
        X = X.__getmatrix__()
        history = model.fit(x=X,y=Y,batch_size=32,epochs=10,verbose=2)
        print("\t\t\tSaving Model:",model.name)
        model.save(path)
        plot_utils.Plot_History(history,model)

    return None

