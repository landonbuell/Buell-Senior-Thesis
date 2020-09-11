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

import ModeUtilities as mode_utils
import SystemUtilities as sys_utils
import NeuralNetworkUtilities as NN_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
    
    # HARD-CODE DIRECTORIES FOR DEVELOPMENT/DEBUGGING
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')
    export = os.path.join(parent,'Output-Data')

    # PRE-PROCESSING FOR PROGRAM
    ProgramSetup = sys_utils.ProgramInitializer(read,model,export,'train-predict',True)    
    FILEOBJECTS,N_classes = ProgramSetup.__Call__()
    exportpath = ProgramSetup.exportpath
    timestart = ProgramSetup.starttime

    # SETUP NEURAL NETWORK MODELS
    NeuralNetwork = NN_utils.NetworkContainer(NN_utils.modelName,N_classes,ProgramSetup.modelpath,
                        NN_utils.inputShapeCNN,NN_utils.inputShapeMLP,ProgramSetup.new_models)
    modelName = NeuralNetwork.name
    
    # DETERMINE WHICH MODE TO RUN PROGRAM
    if ProgramSetup.program_mode == 'train':
        ProgramMode = mode_utils.TrainMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,groupSize=256,n_iters=1)
    elif ProgramSetup.program_mode == 'train-predict':     
        ProgramMode = mode_utils.TrainPredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=False,groupSize=256,n_iters=1,testSize=0.1)
    elif ProgramSetup.program_mode == 'predict':
        ProgramMode = mode_utils.PredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timestart,exportpath=exportpath,
                                            show_summary=True,groupSize=32,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")

    #EXECUTE PROGRAM
    ProgramMode.__Call__(NeuralNetwork)      

    print("=)")
    


        
    