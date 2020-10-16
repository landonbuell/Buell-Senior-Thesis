"""
Landon Buell
PHYS 799
Classifier Cross-Validation
5 October 2020
"""

        #### IMPORTS ####

import sys
import os
import numpy as np

import SystemUtilities as sys_utils
import XValUtilities as XVal_utils

        #### MANIN EXECUTABLE ####

if __name__ == "__main__":

    # HARD-CODE VARIABLES FOR DEVELOPMENT
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis'
    read = os.path.join(parent,'SignalClassifier-CrossValidation','XVal-Target-Data')   
    export = os.path.join(parent,'SignalClassifier-CrossValidation','XVal-Output-Data')
    model = os.path.join(parent,'SignalClassifier-CrossValidation','XVal-Model-Data')

    scriptPath = os.path.join(parent,'SignalClassifier','ClassifierMain')
    scriptName = "ClassifierMAIN.py"

    # HANDLE LOCAL DIRECTORIES
    paths = [read,export,model]
    modelName = "XValCLF-"

 
    # PRE-PROCESSING FOR PROGRAM
    ProgramSetup = sys_utils.ProgramInitializer([read,export,model],modelName)    
    FILEOBJECTS = ProgramSetup.__Call__()
    XValSplits = XVal_utils.CrossValidationSplit(FILEOBJECTS,read,10)
    XValSplits.__Call__()

    # Create Train-Test Splits
    localPaths = [XValSplits.splitsPath,export,model]
    scriptData = [scriptPath,scriptName]
    XVal = XVal_utils.CrossValidator(modelName,XValSplits.K,scriptData,localPaths)
    


    print("=)")
    
