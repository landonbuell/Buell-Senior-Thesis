"""
Landon Buell
Kevin Short
PHYS 799
18 October 2020
"""

        #### IMPORTS ####

import os
import sys
import AnalysisCrossValidationUtilities as utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        #### MAIN EXECUTABLE ####

if __name__ == "__main__":

    # SETUP DIRECTORIES
    parentPath = "C:\\Users\\Landon\\Documents\\GitHub\\Buell-Senior-Thesis\\SignalClassifier-CrossValidation"
    dataPath = os.path.join(parentPath,"XVal-Output-Data")
    modelPath = os.path.join(parentPath,"XVal-Model-Data")

    # MAKE PROGRAM INTIALIZER
    Setup = utils.ProgramSetup(dataPath,modelPath)
    modelNames = Setup.GetModelNames()

    for model in modelNames:
        ModelInst = utils.ModelData(modelPath,model)
        kerasModel = ModelInst.LoadModel()

        """
        Need to figured out methods to access weights.
        Can organize weights into arbitary sized NP array
            Compute Matrix norms-etc.
        """

    print("=)")

