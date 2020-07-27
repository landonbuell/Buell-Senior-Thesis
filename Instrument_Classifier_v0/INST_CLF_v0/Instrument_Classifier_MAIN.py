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
    Program_Initializer = sys_utils.Program_Start(read,model,export,'train-test',False)    
    FILEOBJECTS,N_classes = Program_Initializer.__startup__()
    exportpath = Program_Initializer.exportpath
    timestart = Program_Initializer.starttime

    # SETUP NEURAL NETWORK MODELS
    Neural_Networks = NN_utils.Network_Container(NN_utils.model_names,
        N_classes,Program_Initializer.modelpath,Program_Initializer.new_models)
    model_names = Neural_Networks.__getmodelnames__

    # DETERMINE WHICH MODE TO RUN PROGRAM
    if Program_Initializer.program_mode == 'train':
        exportpath = os.path.join(exportpath,'HISTORY_'+timestart+'.csv')
        Program_Mode = mode_utils.Train_Mode(FILEOBJS=FILEOBJECTS,model_names=model_names,
                                            n_classes=N_classes,exportpath=exportpath,
                                            show_summary=True,n_iters=1)
    elif Program_Initializer.program_mode == 'train-test':     
        exportpath = os.path.join(exportpath,'EVALUATIONS_'+timestart+'.csv')
        Program_Mode =  mode_utils.TrainTest_Mode(FILEOBJS=FILEOBJECTS[:512],model_names=model_names,
                                            n_classes=N_classes,exportpath=exportpath,
                                            show_summary=False,testsize=0.5)
    elif Program_Initializer.program_mode == 'predict':
        exportpath = os.path.join(exportpath,'PREDICTIONS_'+timestart+'.csv')
        Program_Mode = mode_utils.Test_Mode(FILEOBJS=FILEOBJECTS,model_names=model_names,
                                            n_classes=N_classes,exportpath=exportpath,
                                            show_summary=True,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")

    #EXECUTE PROGRAM
    Program_Mode.__call__(Neural_Networks)      

    print("=)")
    


        
    