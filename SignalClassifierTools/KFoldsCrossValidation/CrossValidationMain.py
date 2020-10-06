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
    parent = 'C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifierTools'
    read = os.path.join(parent,'XVal-Target-Data')
    model = os.path.join(parent,'XVal-Model-Data')
    export = os.path.join(parent,'XVal-Output-Data')
    modelName = "XValCLF-"
 
    # PRE-PROCESSING FOR PROGRAM
    ProgramSetup = sys_utils.ProgramInitializer([read,model,export],modelName)    
    FILEOBJECTS = ProgramSetup.__Call__()

    # PREPARE CROSS-VALIDATIONS
    XVal = XVal_utils.CrossValidationSplit(FILEOBJECTS,10)

    print("=)")
    
