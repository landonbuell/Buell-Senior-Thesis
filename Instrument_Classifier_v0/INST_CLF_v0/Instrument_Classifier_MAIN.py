"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import os
import sys

import Mode_Utilities as mode_utils
import System_Utilities as sys_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')

    # PRE-PROCESSING FOR PROGRAM
    Program_Initializer = sys_utils.Program_Start(read,model,mode='train-test')
    MODE = Program_Initializer.program_mode
    FILEOBJECTS,PATH_MAP = Program_Initializer.__startup__()

    if mode == 'train-test':
        Program_Mode = mode_utils.TrainTest_Mode(FILEOBJECTS)

    else:
        print("\n\tError! - Unsupported mode type")



    print("=)")
    


        
    