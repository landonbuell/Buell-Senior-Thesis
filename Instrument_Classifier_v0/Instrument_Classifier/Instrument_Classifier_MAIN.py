"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import Mode_Utilities as mode_utils
import System_Utilities as sys_utils
import Neural_Network_Utilities as NN_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\Instrument_Classifier_v0'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')
    export = os.path.join(parent,'Output-Data')

    # PRE-PROCESSING FOR PROGRAM
    ProgramInitializer = sys_utils.ProgramStart(read,model,export,'train-test',True)    
    FILEOBJECTS,N_classes = ProgramInitializer.__call__()
    exportpath = ProgramInitializer.exportpath
    timestart = ProgramInitializer.starttime

    # SETUP NEURAL NETWORK MODELS
    NeuralNetworks = NN_utils.NetworkContainer(NN_utils.model_names,
        N_classes,ProgramInitializer.modelpath,ProgramInitializer.new_models)
    Model_Names = NeuralNetworks.ModelNames
    
    # DETERMINE WHICH MODE TO RUN PROGRAM
    if ProgramInitializer.program_mode == 'train':
        ProgramMode = mode_utils.TrainMode(FILEOBJS=FILEOBJECTS,model_names=Model_Names,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,n_iters=2)
    elif ProgramInitializer.program_mode == 'train-test':     
        ProgramMode =  mode_utils.TrainTestMode(FILEOBJS=FILEOBJECTS,model_names=Model_Names,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=False,n_iters=4,testsize=0.1)
    elif ProgramInitializer.program_mode == 'predict':
        ProgramMode = mode_utils.TestMode(FILEOBJS=FILEOBJECTS,model_names=Model_Names,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")

    #EXECUTE PROGRAM
    ProgramMode.__CALL__(NeuralNetworks)      

    print("=)")
    


        
    