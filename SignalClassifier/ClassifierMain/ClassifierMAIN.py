"""
Landon Buell
PHYS 799
Instrument Classifier v0
10 June 2020
"""

            #### IMPORTS ####

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ModeUtilities as mode_utils
import SystemUtilities as sys_utils
import NeuralNetworkUtilities as NN_utils

            #### MAIN EXECUTABLE ####

if __name__ == '__main__':
       
    # PRE-PROCESSING FOR PROGRAM
    Args = sys_utils.ArgumentValidator()
    ProgramSetup = sys_utils.ProgramInitializer(Args.GetLocalPaths,Args.GetSystemParams)    
    FILEOBJECTS = ProgramSetup.__Call__()
    (dataPath,exportPath,modelPath) = ProgramSetup.GetLocalPaths
    (modelName,newModel,N_classes,timeStart) = ProgramSetup.GetModelParams

    # SETUP NEURAL NETWORK MODELS
    NeuralNetwork = NN_utils.NetworkContainer(modelName,N_classes,modelPath,
                        NN_utils.inputShapeCNN,NN_utils.inputShapeMLP,newModel)
    NeuralNetwork.SetDecoder(ProgramSetup.GetDecoder)
    
    # DETERMINE WHICH MODE TO RUN PROGRAM
    if ProgramSetup.programMode == 'train':
        ProgramMode = mode_utils.TrainMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,n_iters=4)
    elif ProgramSetup.programMode == 'train-predict':     
        ProgramMode =  mode_utils.TrainPredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,n_iters=4,testSize=0.1)
    elif ProgramSetup.programMode == 'predict':
        ProgramMode = mode_utils.PredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")
        raise ValueError()

    # EXECUTE PROGRAM
    del(ProgramSetup)
    del(Args)

    ProgramMode.__Call__(NeuralNetwork)      

    # FINAL HOUSKEEPING
    ProgramCleanup = sys_utils.ProgramFinisher()
    ProgramCleanup.__Call__(timeStart)
    
    print("=)")
    
