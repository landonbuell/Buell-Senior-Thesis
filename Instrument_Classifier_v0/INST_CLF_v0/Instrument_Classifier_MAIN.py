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
import Neural_Network_Models

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')

    # PRE-PROCESSING FOR PROGRAM
    Program_Initializer = sys_utils.Program_Start(read,model,'train-test',True)
    MODE = Program_Initializer.program_mode
    FILEOBJECTS,N_classes = Program_Initializer.__startup__()

    # SETUP NEURAL NETWORK MODELS
    Neural_Networks = Neural_Network_Models.Network_Models(\)

    if mode == 'train-test':
        
        Program_Mode = mode_utils.Train_Mode(FILEOBJECTS)

    else:
        print("\n\tError! - Unsupported mode type")



    print("=)")
    


        
    