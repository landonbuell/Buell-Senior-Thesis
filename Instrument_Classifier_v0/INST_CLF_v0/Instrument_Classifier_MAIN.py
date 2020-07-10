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
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')

    # PRE-PROCESSING FOR PROGRAM
    Program_Initializer = prog_utils.Program_Start(read,model,mode='train-test')
    MODE = rogram_Initializer.program_mode
    FILEOBJECTS,PATH_MAP = Program_Initializer.__startup__()

    if mode == 'train-test':
        pass

    else:
        print("\n\tError! - Unsupported mode type")



    print("=)")
    


        
    