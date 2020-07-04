"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import numpy as np
import os
import sys

import Component_Utilities as comp_utils

import Program_Utilities as prog_utils
import Plotting_Utilities as plot_utils
import Machine_Learning_Utilities as ML_utils
import Neural_Network_Models

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # ESTABLISH NECESSARY LOCAL PATHS
    PATH_MAP = comp_utils.PATHS()
    PATH_MAP,FILEOBJS,TARGETS = \
       comp_utils.MODELS_FILEOBJS(PATH_MAP,names=['ULTRON','VISION','JARVIS'])
    batch_step = 128      # samples / batch
    print("Files Found:",len(FILEOBJS))

    # ITERATE THROUGH DATA SET BY BATCH
    cntr = 1
    Train = True
    Eval = False
    for I in range(0,len(FILEOBJS),batch_step):  
        print("\tBatch Set:",cntr)
        BATCH = FILEOBJS[I:I+batch_step]    # file objs in batch
        Y = TARGETS[I:I+batch_step]         # Target vector
        comp_utils.TRAIN_on_SET(BATCH,Y,PATH_MAP)
        cntr += 1

        
    