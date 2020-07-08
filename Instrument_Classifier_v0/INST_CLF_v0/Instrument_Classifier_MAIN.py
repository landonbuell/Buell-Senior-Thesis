"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import os
import sys

import Component_Utilities as comp_utils
import Program_Utilities as prog_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # PRE-PROCESSING FOR PROGRAM
    model_names=['JARVIS','VISION','ULTRON']                    # names for models
    PATH_MAP = comp_utils.organize_paths(model_names)           # map of directory paths
    FILEOBJS = prog_utils.Create_Fileobjs(PATH_MAP['DATA'])     # file all fileobjects
    N_classes = prog_utils.np.amax([x.target for x in FILEOBJS]) + 1
    print("Files Found:",len(FILEOBJS))                         # message to user
    TRAIN_FILES,TEST_FILES = prog_utils.split_X(FILEOBJS)       # split train/test     
    comp_utils.create_models(model_names,PATH_MAP,N_classes)    # create network models
    batch_step = 256                                            # samples/batch step

    # ITERATE THROUGH TRAINING FILES
    for I in range (0,len(TRAIN_FILES),batch_step):     # Iter by file batch
        comp_utils.Act_on_Batch(TRAIN_FILES[I:I+batch_step],model_names,
                                PATH_MAP,N_classes,mode='train')
    


        
    