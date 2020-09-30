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
       
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier'
    read = os.path.join(parent,'Target-Data')
    model = os.path.join(parent,'Model-Data')
    export = os.path.join(parent,'Output-Data')
    modelName = "InDevCLF"

    # PRE-PROCESSING FOR PROGRAM
    ProgramSetup = sys_utils.ProgramInitializer([read,model,export],'train-predict',modelName,False)    
    FILEOBJECTS = ProgramSetup.__Call__()
    (dataPath,exportPath,modelPath) = ProgramSetup.GetLocalPaths
    (modelName,newModel,N_classes,timeStart) = ProgramSetup.GetModelParams

    # SETUP NEURAL NETWORK MODELS
    NeuralNetwork = NN_utils.NetworkContainer(modelName,N_classes,modelPath,
                        NN_utils.inputShapeCNN,NN_utils.inputShapeMLP,newModel)
    NeuralNetwork.SetDecoder(ProgramSetup.GetDecoder)
    del(ProgramSetup)
    
    # DETERMINE WHICH MODE TO RUN PROGRAM
    if ProgramSetup.programMode == 'train':
        ProgramMode = mode_utils.TrainMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,n_iters=2)
    elif ProgramSetup.programMode == 'train-predict':     
        ProgramMode =  mode_utils.TrainPredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,n_iters=2,testSize=0.1)
    elif ProgramSetup.programMode == 'predict':
        ProgramMode = mode_utils.PredictMode(FILEOBJS=FILEOBJECTS,modelName=modelName,
                                            n_classes=N_classes,timestamp=timeStart,exportpath=exportPath,
                                            groupSize=256,labels_present=False)
    else:
        print("\n\tError! - Unsupported mode type")
        raise ValueError()

    # EXECUTE PROGRAM
    ProgramMode.__Call__(NeuralNetwork)      

    print("=)")
    


        
    