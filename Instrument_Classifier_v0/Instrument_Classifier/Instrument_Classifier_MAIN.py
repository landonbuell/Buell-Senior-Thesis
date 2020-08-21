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
    ProgramSetup = sys_utils.ProgramInitializer(read,model,export,'train-test',True)    
    FILEOBJECTS,N_classes = ProgramSetup.__Call__()
    exportpath = ProgramSetup.exportpath
    timestart = ProgramSetup.starttime

    # SETUP NEURAL NETWORK MODELS
    NeuralNetwork = NN_utils.NetworkContainer(NN_utils.modelName,N_classes,exportpath,
                        NN_utils.inputShapeCNN,NN_utils.inputShapeMLP,ProgramSetup.new_models)
    modelName = NeuralNetwork.name

    print(NeuralNetwork.MODEL.summary())
    
    # DETERMINE WHICH MODE TO RUN PROGRAM
    if ProgramSetup.program_mode == 'train':
        ProgramMode = mode_utils.TrainMode(FILEOBJS=FILEOBJECTS,modelNames=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,n_iters=2)
    elif ProgramSetup.program_mode == 'train-test':     
        ProgramMode =  mode_utils.TrainTestMode(FILEOBJS=FILEOBJECTS,modelNames=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=False,n_iters=4,testsize=0.1)
    elif ProgramSetup.program_mode == 'predict':
        ProgramMode = mode_utils.TestMode(FILEOBJS=FILEOBJECTS,modelNames=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")

    #EXECUTE PROGRAM
    ProgramMode.__CALL__(NeuralNetworks)      

    print("=)")
    


        
    