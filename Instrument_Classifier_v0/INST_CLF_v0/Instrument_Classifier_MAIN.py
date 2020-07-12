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
import Neural_Network_Utilities as NN_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')

    # PRE-PROCESSING FOR PROGRAM
    Program_Initializer = sys_utils.Program_Start(read,model,'train-test',True)   
    FILEOBJECTS,N_classes = Program_Initializer.__startup__()

    # SETUP NEURAL NETWORK MODELS
    Neural_Networks = NN_utils.Network_Container(NN_utils.model_names,
        N_classes,Program_Initializer.modelpath,Program_Initializer.new_models)
    model_names = Neural_Networks.__getmodelnames__

    # DETERMINE WHICH MODE TO RUN PROGRAM IN
    if Program_Initializer.program_mode == 'train':
        Program_Mode = mode_utils.Train_Mode(FILEOBJECTS,model_names)
    elif Program_Initializer.program_mode == 'train-test':     
        Program_Mode =  mode_utils.TrainTest_Mode(FILEOBJECTS,model_names)
    elif Program_Initializer.program_mode == 'predict':
        Program_Mode = mode_utils.Test_Mode(FILEOBJECTS,model_names)
    else:
        print("\n\tError! - Unsupported mode type")

    #EXECUTE PROGRAM
    Program_Mode.__call__(Neural_Networks)      

    print("=)")
    


        
    